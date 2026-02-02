# 模型定义在此
from transformers import PretrainedConfig
import math
import torch
import torch.nn.functional as F
from torch import nn

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
def apply_rotary_emb(q,k,cos,sin):
    return q, k

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
            past_key_value,
            attention_mask,
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
            # 使用Flash Attention。
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

# if __name__ == '__main__':
