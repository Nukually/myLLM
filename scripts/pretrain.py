import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from model.myllm import MyLLM, ModelConfig
import torch.optim as optim
from tqdm import tqdm
import wandb

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'text' in item:
                        self.data.append(item['text'])
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        # Mask pad tokens in labels so we don't calculate loss for them
        # Assuming pad_token_id is 0 (need to verify with tokenizer)
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def train():
    # Configuration
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 1
    MAX_LENGTH = 340
    SAVE_DIR = "checkpoints"
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialize Tokenizer
    # Assuming tokenizer is already trained and saved in 'model/tokenizer.json'
    tokenizer_path = "model/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print("Tokenizer not found! Please train tokenizer first.")
        return

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    
    # Initialize Dataset & DataLoader
    data_path = "dataset/pretrain_hq.jsonl"
    if not os.path.exists(data_path):
         # Create dummy data for testing if file is empty/missing
        print(f"Warning: {data_path} not found. Creating dummy data.")
        with open(data_path, 'w') as f:
            for i in range(100):
                f.write(json.dumps({"text": f"This is a sample training sentence {i}."}) + "\n")
    
    dataset = PretrainDataset(data_path, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    config = ModelConfig(
        vocab_size=len(tokenizer),
        n_layers=8,        # Small config for testing
        hidden_size=512,
        n_heads=8,
        n_kvheads=2,
        max_pe=MAX_LENGTH
    )
    model = MyLLM(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to {device}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    model.train()
    global_step = 0
    total_loss = 0
    
    progress_bar = tqdm(range(len(dataloader) * NUM_EPOCHS))
    
    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            total_loss += loss.item()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
                global_step += 1
                
                if global_step % 100 == 0:
                    print(f"\nStep {global_step}: Loss = {total_loss / 100:.4f}")
                    total_loss = 0
                    
        # Save checkpoint
        checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train()
