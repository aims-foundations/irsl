import pandas as pd
import torch
from tqdm import tqdm
from torch.distributions import Bernoulli
from torchmetrics import AUROC
auroc = AUROC(task="binary")
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
torch.manual_seed(0)
gpuid = 7

input_df = pd.read_csv('gsm_hard_easy_200.csv')
data = torch.tensor(input_df.values, dtype=torch.float32)
N, P = data.shape
subsets = []
leftovers = []
train_percentage = 0.86
N_train = int(N*train_percentage)
lr = 0.1
l1_lambda = 0.1
n_epochs = 10000
n_epochs_refit = 10000

for i in range(P):
    # Mask to remove i-th column
    mask = torch.ones(P, dtype=torch.bool)
    mask[i] = False
    subset = data[:, mask] # shape: (N, P-1)
    # add a one column to the subset
    subset = torch.cat((subset, torch.ones(N, 1)), dim=1) # shape: (N, P)
    leftover = data[:, i] # shape: (N,)

    subsets.append(subset.unsqueeze(2)) # shape: (N, P-1, 1)
    leftovers.append(leftover.unsqueeze(1)) # shape: (N, 1)

inputs = torch.cat(subsets, dim=2) # shape: (N, P-1, P)
labels = torch.cat(leftovers, dim=1) # shape: (N, P)

def soft_thresholding(x, alpha):
    return torch.sign(x) * torch.clamp(torch.abs(x) - alpha, min=0.0)

class RefitLogisticRegression(torch.nn.Module):
    def __init__(self, P, B):
        super().__init__()
        self.linear = torch.nn.Linear(P, B, bias=False)

    def forward(self, inputs):
        W = self.linear.weight  # shape: (P, B)
        logits = torch.einsum('ijp,jp->ip', inputs, W) # shape: (N_train, B)
        return torch.sigmoid(logits)

step_size = 1024
Ws = []
for i in tqdm(range(0, P, step_size)):
    B = min(step_size, P - i)
    inputs_ = inputs[:, :, i:i+B].to(f'cuda:{gpuid}') # shape: (N, P-1, B)
    labels_ = labels[:, i:i+B].to(f'cuda:{gpuid}') # shape: (N, B)

    indices = torch.randperm(N)
    train_indices = indices[:N_train]
    test_indices = indices[N_train:]
    inputs_train = inputs_[train_indices, :, :]
    inputs_test = inputs_[test_indices, :, :]
    labels_train = labels_[train_indices, :]
    labels_test = labels_[test_indices, :]

    W = torch.randn((P, B), device=f'cuda:{gpuid}', requires_grad=False)
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        logits = torch.einsum('ijp,jp->ip', inputs_train, W) # shape: (N_train, B)
        probs = torch.sigmoid(logits)
        
        error = probs - labels_train  # shape: (N_train, B)
        grad = torch.einsum('ijp,ip->jp', inputs_train, error) / N_train # shape: (P, B)
        W_temp = W - lr * grad
        W = soft_thresholding(W_temp, lr * l1_lambda)
        
        if epoch % 10 == 0:
            loss = -Bernoulli(probs=probs).log_prob(labels_train).mean()
            reg_term = l1_lambda * torch.norm(W, p=1, dim=0).mean()
            # pbar.set_postfix({"loss": loss.item(), "reg_term": reg_term})
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, L1 penalty: {reg_term.item():.4f}")

    selected_idx = (W.abs() > 1e-3) # shape: (P, B)
    selected_idx = selected_idx.unsqueeze(0).expand(N_train, -1, -1) # shape: (N, P, B)
    inputs_train_selected = inputs_train * selected_idx # shape: (N, P, B)
    
    refit_model = RefitLogisticRegression(P, B).to(f'cuda:{gpuid}')
    optimizer = torch.optim.Adam(refit_model.parameters(), lr=0.05)
    loss_fn = torch.nn.BCELoss()
    for epoch in range(n_epochs_refit):
        optimizer.zero_grad()
        probs = refit_model(inputs_train_selected)
        loss = loss_fn(probs, labels_train)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Refit loss: {loss.item():.4f}")

    W = refit_model.linear.weight.data.squeeze()
    Ws.append(W)
    torch.save(W, f'W_{i}.pt')
        
    logits_train = torch.einsum('ijp,jp->ip', inputs_train, W) # shape: (N, B)
    final_auc_train = auroc(torch.sigmoid(logits_train), labels_train)
    logits_test = torch.einsum('ijp,jp->ip', inputs_test, W)
    final_auc_test = auroc(torch.sigmoid(logits_test), labels_test)
    print(f"final_auc_train: {final_auc_train}; final_auc_test: {final_auc_test}")

Ws = torch.cat(Ws, dim=1) # shape: (P, P)
non_zero_mask = (Ws != 0) & (Ws.T != 0)
Ws_avg = 0.5 * (Ws + Ws.T)
Ws = torch.where(non_zero_mask, Ws_avg, torch.zeros_like(Ws))
Ws[torch.arange(Ws.size(0)), torch.arange(Ws.size(1))] = 0
torch.save(Ws, 'W.pt')
W_df = pd.DataFrame(Ws.cpu().numpy(), columns=input_df.columns)
W_df.to_csv('W.csv', index=False)

non_zero_count = (Ws != 0).sum().item()
total_elements = Ws.numel()
percent_non_zero = 100.0 * non_zero_count / total_elements
print(f"Non-zero elements: {non_zero_count} ({percent_non_zero:.2f}%)")

# Visualize the matrix
plt.figure(figsize=(8, 6))
plt.imshow(Ws.cpu().numpy(), cmap='viridis')
plt.title("Matrix Visualization")
plt.colorbar(label="Value")
plt.savefig("matrix_visualization.png", dpi=300, bbox_inches="tight")
plt.close()