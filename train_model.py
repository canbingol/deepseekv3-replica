import torch

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

import os
from itertools import islice
from warnings import filterwarnings

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import sentencepiece as spm
from data import train_loader, val_loader
from model import ModelArgs, Transformer

filterwarnings('ignore')

args = ModelArgs()
EPOCH = 1
MAX_STEP = 500
LR = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
""" if torch.backends.mps.is_available():
    device = 'mps' """
print(f'current device is {device}')


model = Transformer(ModelArgs)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

moe_params = sum(
    p.numel() for n, p in model.named_parameters()
    if any(key in n for key in ['experts', 'gate', 'shared_experts']) and p.requires_grad
)
tokenizer_path = '/kaggle/working/turna_noe/tokenizer.model' if os.path.exists('https://github.com/canbingol/turna_noe/edit/main/train_model.py') else 'tokenizer.model'
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(tokenizer_path)
print(f"tokenizer path: {tokenizer_path}")
print(f"tokenizer vocab size: {tokenizer.vocab_size()}")
non_moe_params = trainable_params - moe_params

moe_inter_dim = args.moe_inter_dim
dim = args.dim
topk = args.n_activated_experts
n_shared = args.n_shared_experts

single_expert_params = (dim * moe_inter_dim) * 2 + (dim * moe_inter_dim)  # w1, w2, w3
active_expert_params = topk * single_expert_params
shared_expert_params = (dim * (moe_inter_dim * n_shared)) * 2 + (dim * (moe_inter_dim * n_shared))
inference_active_params = non_moe_params + active_expert_params + shared_expert_params

print(f"{'Toplam Parametre Sayısı':<40}: {total_params:,}")
print(f"{'Eğitilebilir Parametre Sayısı':<40}: {trainable_params:,}")
print(f"{'MoE Parametrelerinin Tamamı':<40}: {moe_params:,}")
print(f"{'Sadece Inference için Aktif Kullanılan Tahmini Parametre':<40}: {inference_active_params:,}")


print(f"train data len : {len(train_loader)}")
print(f"val data len : {len(val_loader)}")

prompt = "Merhaba"

optimizer = torch.optim.Adam(model.parameters(),lr=LR)
scheduler = StepLR(optimizer, step_size=1, gamma=0.94)  # her epoch sonunda lr *= 0.9

len_train_data = len(train_loader)
len_val_data = len(val_loader)

train_losses = []
val_losses = []

n_val_sample = 10

import torch.nn.functional as F


def sample_from_model(input_ids, max_new_tokens, device='cuda', temperature=1.0, top_k=50, top_p=0.95):
    model.eval()
    input_ids = input_ids.to(device)

    for _ in range(max_new_tokens):
        seq_len = input_ids.size(1)
        with torch.no_grad():
            logits = model(input_ids, start_pos=seq_len - 1, return_gate_info=False)

        next_token_logits = logits[:, -1, :] / temperature  # Temperature scaling

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, next_token_logits.size(-1))
            values, _ = torch.topk(next_token_logits, top_k)
            min_threshold = values[:, -1].unsqueeze(-1)
            next_token_logits = torch.where(next_token_logits < min_threshold, torch.full_like(next_token_logits, -float('Inf')), next_token_logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('Inf'))

        # Sampling
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    return input_ids


def compute_balance_loss(scores, batch_size, seq_len, n_experts,topk_indices, topk=2, alpha=1e-4):
    s  = scores.view(batch_size,seq_len,n_experts )

    f = torch.zeros(n_experts, device=s.device)
    for i in range(n_experts):
        f[i] = (topk_indices == i).sum().float() / (batch_size * seq_len * topk)

    # s'_i,t 
    s_sum = s.sum(dim=-1,keepdim=True)
    s_norm = s / s_sum

    #p_i
    p_i = s_norm.mean(dim=1).mean(dim=0)

    # L_bal
    l_bal = alpha * (f * p_i).sum()
    return l_bal
 
def eval_model(epoch: int, step_id: int):
    args.train = False
    model.eval()
    val_iter = islice(val_loader, n_val_sample)
    total_val_loss = 0
    avg_val_loss_so_far = 0

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(val_iter):

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch, start_pos=0, return_gate_info=False)

            loss = F.cross_entropy(logits.view(-1, args.vocab_size), target_batch.view(-1))
            total_val_loss += loss.item()
            
            avg_val_loss_so_far = total_val_loss / (i + 1)
            val_losses.append(avg_val_loss_so_far)

            with open('logs/val_loss.txt', 'a') as f:
                f.write(f'{step_id},{avg_val_loss_so_far}\n')


    avg_val_loss = total_val_loss / n_val_sample
    model.train()
    args.train = True
    return avg_val_loss

for epoch in range(EPOCH):
    if tokenizer.vocab_size() != model.embed.weight.shape[0]:
        print(f"tokenizer.vocab_size(){tokenizer.vocab_size()} ile  model.embed.weight.shape { model.embed.weight.shape[0]} eşit değil")
        break
    args.train = True
    model.train()
    total_train_loss = 0
    count = 0

    print(f"\n[Epoch {epoch+1}/{EPOCH}] Training started | Learning rate: {scheduler.get_last_lr()[0]:.6f}")

    for i, (input_batch, target_batch) in enumerate(train_loader):
        step_id = epoch * len_train_data + i + 1

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        logits, scores, topk_indices = model(input_batch, start_pos=0, return_gate_info=True)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[Error] NaN or Inf detected in logits at step {step_id}")
            print("Logits sample:", logits[0])
            exit()

        loss_main = F.cross_entropy(logits.view(-1, args.vocab_size), target_batch.view(-1))
        l_bal = compute_balance_loss(scores=scores,
                                     batch_size=input_batch.size(0),
                                     seq_len=input_batch.size(1),
                                     n_experts=args.n_routed_experts,
                                     topk_indices=topk_indices,
                                     topk=args.n_activated_experts,
                                     alpha=1e-4)
        
        loss = loss_main + l_bal
        if torch.isnan(loss):
            print(f"[Error] NaN loss at step {step_id}")
            print("Target:", target_batch)
            exit()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        avg_loss = total_train_loss / (i + 1)
        train_losses.append(avg_loss)

        if (i + 1) % 10 == 0 or (i + 1) == len_train_data:
            count = count +1
            avg_val_loss = eval_model(epoch, step_id)
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Epoch {epoch+1} | Step {i+1:>6}/{len_train_data}] "
                  f"Train Loss: {avg_loss:.4f}| Val Loss: {avg_val_loss:.4f} | L_bal: {l_bal    } | LR: {current_lr:.6f}")

        if (i+1) % 100 == 0 or (i + 1) == len_train_data:
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long) 
            generated_ids = sample_from_model(input_ids, max_new_tokens=50, device=device)
            print("Üretilen metin:\n", tokenizer.decode(generated_ids[0].tolist()))

        with open('logs/train_loss.txt', 'a') as f:
            f.write(f'{step_id},{avg_loss}\n')

        if (i + 1) % 10_000 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
            }, f"checkpoint_epoch_{epoch+1}_step_{i+1}.pt")

        if step_id == MAX_STEP:
            break

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = val_losses[-1]
    scheduler.step()

    print(f"[Epoch {epoch+1} completed] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses,
        'val_loss': val_losses,
    }, f"checkpoint_epoch_{epoch+1}.pt")



fig, axes = plt.subplots(1, 2, figsize=(14, 5))  

# Train loss grafiği
axes[0].plot(train_losses, label='Train Loss',linestyle='-')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Train Loss')
axes[0].legend()
axes[0].grid(True)

# Validation loss grafiği
axes[1].plot(val_losses, label='Validation Loss', marker='x', color='orange')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Loss')
axes[1].set_title('Validation Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("loss_curve_side_by_side.png")
plt.show()
