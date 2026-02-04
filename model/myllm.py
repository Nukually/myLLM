# 模型定义在此
from typing import Optional, Tuple, List
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

class ModelConfig(PretrainedConfig):
    model_type="myllm"
    def __init__(
            self,
            dropout: float = 0.0,
            n_heads:int=16,
            n_kvheads:int=8,
            n_layers:int=8,
            hidden_size: int = 768,
            flash_attn: bool = True, # 是否使用Flash Attention
            vocab_size=6400,
            multiple_of:int=64,
            # PE
            max_pe: int = 32768,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            # TODO:MoE
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout=dropout
        self.n_heads=n_heads
        self.n_kvheads=n_kvheads
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.flash_attn=flash_attn
        self.vocab_size=vocab_size
        self.max_pe=max_pe
        self.rms_norm_eps=rms_norm_eps
        self.rope_theta=rope_theta
        self.multiple_of=multiple_of
        self.rope_scaling=None

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
    
# 满足GQA场景下一组KV服务多个Q头的需要
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """input->[bs,sl,n_kvheads,head_dim]"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

# TODO:RoPE
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0,rope_base:float=1e6,rope_scaling=None):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，长度为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取x的维度数
    ndim = x.ndim
    # 断言，确保1在x的维度范围内
    assert 0 <= 1 < ndim
    # 断言，确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # 将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self,args:ModelConfig):
        super().__init__()
        # 参数设置
        self.n_kvheads=args.n_heads if args.n_kvheads is None else args.n_kvheads
        assert args.n_heads % args.n_kvheads == 0
        model_parallel_size = 1
        self.n_heads_local=args.n_heads // model_parallel_size
        self.n_kvheads_local=self.n_kvheads //  model_parallel_size
        self.n_rep=self.n_heads_local // self.n_kvheads_local
        self.head_dim=args.hidden_size//args.n_heads
        # 权重矩阵
        self.wq = nn.Linear(args.hidden_size, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.n_kvheads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.hidden_size, self.n_kvheads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.hidden_size, bias=False)
        # dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(
            self,
            x:torch.Tensor,
            pe,
            *,
            past_key_value=None,
            attention_mask=None,
            use_cache=False,
    ):
        bs,sl,_=x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq=xq.view(bs,sl,self.n_heads_local,self.head_dim)
        xk=xk.view(bs,sl,self.n_kvheads_local,self.head_dim)
        xv=xv.view(bs,sl,self.n_kvheads_local,self.head_dim)
        cos,sin=pe
        xq,xk=apply_rotary_emb(xq,xk,cos,sin)
        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        # 支持选用flash attention. 注意Flash Attention 在有 KV cache 或自定义 mask 时可能不兼容
        if self.flash and (past_key_value is None) and (attention_mask is None):
            # 使用Flash Attention
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores[:, :, :, -sl:] += torch.triu(torch.full((sl, sl), float("-inf"), device=scores.device), diagonal=1)
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bs, sl, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
class DecoderLayer(nn.Module):
    def __init__(self,layer_id:int,args:ModelConfig):
        super().__init__()
        self.n_heads=args.n_heads
        self.hidden_size=args.hidden_size
        self.head_dim=args.hidden_size//args.n_heads
        self.attention=Attention(args)
        self.ffn=MLP(dim=args.hidden_size,hidden_dim=None,multiple_of=args.multiple_of,dropout=args.dropout)
        self.layer_id=layer_id
        self.attention_norm=RMSNorm(dim=args.hidden_size,eps=args.rms_norm_eps)
        self.ffn_norm=RMSNorm(dim=args.hidden_size,eps=args.rms_norm_eps)

    def forward(self, x, pe, past_key_value=None, use_cache=False, attention_mask=None):
        res=x
        x,present_kv=self.attention(self.attention_norm(x),pe,past_key_value=past_key_value,attention_mask=attention_mask,use_cache=use_cache)
        x+=res
        x=x+self.ffn(self.ffn_norm(x))
        return x,present_kv

# 不带输出头的
class BaseModel(nn.Module):
    def __init__(self,args:ModelConfig):
        super().__init__()
        self.args=args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.dropout=nn.Dropout(args.dropout)
        self.layers=nn.ModuleList([DecoderLayer(l,args) for l in range(self.n_layers)])
        self.norm=RMSNorm(args.hidden_size,eps=args.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=args.hidden_size // args.n_heads,
                                                    end=args.max_pe, rope_base=args.rope_theta,
                                                    rope_scaling=args.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def forward(self,input_ids=None,attention_mask: Optional[torch.Tensor]=None,past_kv=None,use_cache=False,**kwargs):
        bs,sl=input_ids.shape
        past_kv = past_kv or [None] * len(self.layers)
        start_pos = past_kv[0][0].shape[1] if past_kv[0] is not None else 0
        x = self.dropout(self.embed_tokens(input_ids))
        pe = (
            self.freqs_cos[start_pos:start_pos + sl],
            self.freqs_sin[start_pos:start_pos + sl]
        )
        presents = []
        for layer_idx,(layer,past_kv_single) in enumerate(zip(self.layers,past_kv)):
            x,present=layer(
                x,
                pe,
                past_key_value=past_kv_single,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        x=self.norm(x)
        return x,presents


class MyLLM(PreTrainedModel):
    config_class = ModelConfig
    _no_split_modules = ["DecoderLayer"]

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = BaseModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_kv=past_key_values,
            use_cache=use_cache,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs[1],
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,
        )

    
if __name__ == '__main__':
    # Test code
    print("Testing MyLLM Model...")
    
    # 1. Initialize Config and Model
    config = ModelConfig(
        n_layers=2,        # Small model for testing
        hidden_size=64,
        n_heads=4,
        n_kvheads=2,
        vocab_size=100
    )
    model = MyLLM(config)
    print("Model initialized successfully.")
    
    # 2. Test Forward Pass (No Cache)
    input_ids = torch.randint(0, config.vocab_size, (2, 10)) # Batch size 2, Seq len 10
    outputs = model(input_ids=input_ids)
    print(f"Forward pass output logits shape: {outputs.logits.shape} (Expected: [2, 10, 100])")
    assert outputs.logits.shape == (2, 10, 100)
    
    # 3. Test Loss Calculation
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)
    print(f"Loss value: {outputs.loss.item()}")
    assert outputs.loss is not None
    
    # 4. Test KV Cache (Generation Step)
    # First step
    outputs_1 = model(input_ids=input_ids, use_cache=True)
    past_kv = outputs_1.past_key_values
    print(f"Past KV length: {len(past_kv)}") # Should be n_layers
    
    # Next step (append one token)
    next_token = torch.randint(0, config.vocab_size, (2, 1))
    outputs_2 = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
    print(f"Next step logits shape: {outputs_2.logits.shape} (Expected: [2, 1, 100])")
    assert outputs_2.logits.shape == (2, 1, 100)
    
    print("All tests passed!")