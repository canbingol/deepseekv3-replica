import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import ModelArgs
import os
import sentencepiece as spm
import math

from warnings import filterwarnings
filterwarnings('ignore')


tokenizer = spm.SentencePieceProcessor()
tokenizer.load("tokenizer.model")

args = ModelArgs()
max_len = args.max_seq_len
pad_token = tokenizer.pad_id()


data_path = ModelArgs.dataset_path

with open(data_path, "r", encoding="utf-8") as f:
    raw_data = f.readlines()
print(f'len raw data {len(raw_data)}')
train_ratio = .85
train_len = int(len(raw_data) * train_ratio)

train_data = raw_data[:train_len]
val_data = raw_data[train_len:]
print(f'data path : {ModelArgs.dataset_path}')

class language_model_dataset(Dataset):

    def __init__(self, texts, tokenizer, max_len, stride):
        super().__init__()
        self.inputs = []
        self.targets = []
        input_tokens = []
        for i, text in enumerate(texts):
            tokens = tokenizer.encode(text)
            token_list = tokens
            input_tokens.append(token_list)  

            
        for chunk in input_tokens: 

            len_ratio = math.ceil(len(chunk)/stride )   
            chunk_len =  len(chunk)
            chunk_len +=  len_ratio * stride - chunk_len

            for i in range(0, chunk_len - max_len+1, stride):
                input_chunk = chunk[i: i+max_len]
                target_chunk = chunk[1+ i: 1+ i+max_len]
                
                # truncating
                input_chunk = input_chunk[:max_len]
                target_chunk = target_chunk[:max_len]

                # padding
                input_chunk += [pad_token] * (max_len - len(input_chunk))
                target_chunk += [pad_token] * (max_len - len(target_chunk)) 

                input_chunk = input_chunk[:max_len]
                target_chunk = target_chunk[:max_len]

                self.inputs.append(torch.tensor(input_chunk, dtype=torch.long))
                self.targets.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

print()


def create_dataloader(text: list, batch_size: int, drop_last: bool,
                      max_len: int, stride: int, shuffle: bool, tokenizer):
    
    dataset = language_model_dataset(text, tokenizer,max_len, stride)
    dataloader = DataLoader(
        dataset  =dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last=drop_last,
    )
    
    return dataloader

train_loader = create_dataloader(train_data,args.max_batch_size,args.drop_last,
                                 args.max_seq_len,args.max_seq_len,args.shuffle ,tokenizer)


val_loader = create_dataloader(val_data,args.max_batch_size,args.drop_last,
                                 args.max_seq_len,args.max_seq_len,args.shuffle, tokenizer )


def save_sample_batch(dataloader, tokenizer, filename: str, num_samples: int = 2):
    with open(filename, "w", encoding="utf-8") as f:
        for i, (inputs, targets) in enumerate(dataloader):
            for j in range(min(num_samples, inputs.size(0))):
                input_ids = inputs[j].tolist()
                target_ids = targets[j].tolist()

                input_text = tokenizer.decode(input_ids)
                target_text = tokenizer.decode(target_ids)

                f.write(f"Sample {i * num_samples + j + 1}\n")
                f.write(f"Input IDs  : {input_ids}\n")
                f.write(f"Input Text : {input_text}\n")
                f.write(f"Target IDs : {target_ids}\n")
                f.write(f"Target Text: {target_text}\n")
                f.write("\n" + "=" * 80 + "\n\n")
            break  # Sadece ilk batch'ten örnek al


save_sample_batch(train_loader, tokenizer, "logs/train_sample.txt")
save_sample_batch(val_loader, tokenizer, "logs/val_sample.txt")
print(f"len train data {len(train_loader)}")
print(f"len val data {len(val_loader)}")
