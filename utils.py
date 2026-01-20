import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles
bundles.icml2024()
from tqdm import tqdm
import torch
torch.manual_seed(0)
from torch.distributions import Bernoulli
import numpy as np
from torch.optim import LBFGS
from collections import defaultdict

MODEL2BATCH = {
    '4M': 32, # batch_size=32, gpus=8
    '6M': 32,
    '8M': 32,
    '10M': 32,
    '14M': 32,
    '16M': 32,
    '20M': 64,
    '60M': 96,
    '90M': 160,
    '150M': 192,
    '300M': 320,
    '530M': 448,
    '750M': 576,
    '1B': 704
}
MODEL2PARA = {
    '4M': 3_744_832,
    '6M': 6_010_464,
    '8M': 8_538_240,
    '10M': 9_900_432,
    '12M': 12_066_600,
    '14M': 14_380_224,
    '16M': 16_004_560,
    '20M': 19_101_888,
    '60M': 57_078_144,
    '90M': 97_946_640,
    '150M': 151_898_880,
    '300M': 319_980_544,
    '530M': 530_074_944,
    '750M': 681_297_408,
    '1B': 1_176_832_000
}

def calculate_flops(model_size: str, step: int) -> float:
    SEQUENCE_LENGTH = 2048
    n = float(MODEL2PARA[model_size])
    d = float(MODEL2BATCH[model_size]) * float(step) * float(SEQUENCE_LENGTH)
    return n * d * 6.0

def recursive_defaultdict():
        return defaultdict(recursive_defaultdict)






def calibrate_1pl_z(
    resmat: torch.Tensor,
    device: str,
    n_thetas_nuisance: int = 150,
    eps: float = 1e-6,
    phi: float = 10.0,
    batch_size: int = 50000,
    loss_kind: str = "beta"
) -> np.ndarray:
    resmat = resmat.to(device)
    n_test_takers, n_items = resmat.shape
    thetas_nuisance = torch.randn(n_thetas_nuisance, n_test_takers, device=device)
    phi_tensor = torch.tensor(phi, device=device)

    if loss_kind == "beta":
        def compute_loss(response_batch, mu, mask):
            y = response_batch.expand(n_thetas_nuisance, -1, -1).clamp(eps, 1 - eps)
            return beta_nll(y[mask], mu[mask], phi_tensor).mean()
    elif loss_kind == "binary":
        def compute_loss(response_batch, mu, mask):
            y = response_batch.expand(n_thetas_nuisance, -1, -1)
            return -Bernoulli(probs=mu[mask]).log_prob(y[mask]).mean()
    else:
        raise ValueError(f"Unknown loss_kind: {loss_kind}")

    optimized_zs = []
    for start in tqdm(range(0, n_items, batch_size)):
        end = min(start + batch_size, n_items)
        response_batch = resmat[:, start:end]
        z_batch = torch.randn(end - start, requires_grad=True, device=device)
        optim_z = LBFGS([z_batch], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

        def closure_z():
            optim_z.zero_grad()
            mask = ~torch.isnan(response_batch).expand(n_thetas_nuisance, -1, -1)
            mu = torch.sigmoid(thetas_nuisance[:, :, None] + z_batch[None, None, :])
            loss = compute_loss(response_batch, mu, mask)
            loss.backward()
            return loss

        z_opt = trainer([z_batch], optim_z, closure_z, verbose=True)[0].detach()
        optimized_zs.append(z_opt)

    return torch.cat(optimized_zs).cpu().numpy()

def calibrate_1pl_theta(
    resmat: torch.Tensor,
    device: str,
    zs: torch.Tensor,
    eps: float = 1e-6,
    phi: float = 10.0,
    loss_kind: str = "beta"
) -> np.ndarray:
    resmat, zs = resmat.to(device), zs.to(device)
    n_test_takers, n_items = resmat.shape
    phi_tensor = torch.tensor(phi, device=device)

    if loss_kind == "beta":
        def compute_loss(y, mu, mask):
            y = y.clamp(eps, 1 - eps)
            return beta_nll(y[mask], mu[mask], phi_tensor).mean()
    elif loss_kind == "binary":
        def compute_loss(y, mu, mask):
            return -Bernoulli(probs=mu[mask]).log_prob(y[mask]).mean()
    else:
        raise ValueError(f"Unknown loss_kind: {loss_kind}")
    
    thetas = torch.randn(n_test_takers, requires_grad=True, device=device)
    optim_theta = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

    def closure_theta():
        optim_theta.zero_grad()
        mask = ~torch.isnan(resmat)
        mu = torch.sigmoid(thetas[:, None] + zs[None, :])
        loss = compute_loss(resmat, mu, mask)
        loss.backward()
        return loss

    thetas = trainer([thetas], optim_theta, closure_theta)[0].detach()
    return thetas.cpu().numpy()






def cat_beta_1pl(ys, zs, device, budget=50):
    return _cat_core(
        ys=ys, zs=zs, device=device,
        estimator_fn=_wrap_estimator_fn_beta_1pl, select_next_fn=_select_next_1pl, discris=None, budget=budget
    )

def cat_beta_2pl(ys, discris, zs, device, budget=50):
    return _cat_core(
        ys=ys, zs=zs, device=device,
        estimator_fn=_wrap_estimator_fn_beta_2pl, select_next_fn=_select_next_2pl, discris=discris, budget=budget
    )

def cat_binary_1pl(ys, zs, device, budget=50):
    return _cat_core(
        ys=ys, zs=zs, device=device,
        estimator_fn=_wrap_estimator_fn_binary_1pl, select_next_fn=_select_next_1pl, discris=None, budget=budget
    )

def cat_binary_2pl(ys, discris, zs, device, budget=50):
    return _cat_core(
        ys=ys, zs=zs, device=device,
        estimator_fn=_wrap_estimator_fn_binary_2pl, select_next_fn=_select_next_2pl, discris=discris, budget=budget
    )

def _wrap_estimator_fn_beta_1pl(theta, asked_y, asked_discri, asked_z, device):
    return _estimate_theta_beta_1pl(theta, asked_y, asked_z, device)

def _wrap_estimator_fn_beta_2pl(theta, asked_y, asked_discri, asked_z, device):
    return _estimate_theta_beta_2pl(theta, asked_y, asked_discri, asked_z, device)

def _wrap_estimator_fn_binary_1pl(theta, asked_y, asked_discri, asked_z, device):
    return _estimate_theta_binary_1pl(theta, asked_y, asked_z, device)

def _wrap_estimator_fn_binary_2pl(theta, asked_y, asked_discri, asked_z, device):
    return _estimate_theta_binary_2pl(theta, asked_y, asked_discri, asked_z, device)

def _estimate_theta_beta_1pl(theta, asked_ys, asked_zs, device, eps=1e-6):
    asked_ys = asked_ys.clamp(min=eps, max=1.0 - eps)
    asked_zs = torch.as_tensor(asked_zs, device=device, dtype=torch.float)
    return _estimate_theta_generic(
        theta, asked_ys, device,
        logits_fn=lambda th: th + asked_zs,
        loss_kind="beta"
    )

def _estimate_theta_beta_2pl(theta, asked_ys, asked_discris, asked_zs, device, eps=1e-6):
    asked_discris = torch.as_tensor(asked_discris, device=device, dtype=torch.float)
    asked_zs      = torch.as_tensor(asked_zs,      device=device, dtype=torch.float)
    asked_ys = asked_ys.clamp(min=eps, max=1.0 - eps)
    return _estimate_theta_generic(
        theta, asked_ys, device,
        logits_fn=lambda th: asked_discris * (th - asked_zs), 
        loss_kind="beta"
    )

def _estimate_theta_binary_1pl(theta, asked_ys, asked_zs, device):
    asked_zs = torch.as_tensor(asked_zs, device=device, dtype=torch.float)
    return _estimate_theta_generic(
        theta, asked_ys, device,
        logits_fn=lambda th: th + asked_zs,
        loss_kind="binary"
    )

def _estimate_theta_binary_2pl(theta, asked_ys, asked_discris, asked_zs, device):
    asked_discris = torch.as_tensor(asked_discris, device=device, dtype=torch.float)
    asked_zs      = torch.as_tensor(asked_zs,      device=device, dtype=torch.float)
    return _estimate_theta_generic(
        theta, asked_ys, device,
        logits_fn=lambda th: asked_discris * (th - asked_zs),
        loss_kind="binary"
    )

def _estimate_theta_generic(
    theta, asked_ys, device,
    logits_fn,
    loss_kind, # "beta" or "binary"
    phi=10.0, eps=1e-6, lr=0.1
):
    asked_ys = torch.as_tensor(asked_ys, device=device, dtype=torch.float)
    theta = theta.clone().requires_grad_(True)
    optim = torch.optim.LBFGS([theta], lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    phi_t = torch.as_tensor(phi, device=device, dtype=torch.float)
    
    def closure():
        optim.zero_grad()
        logits = logits_fn(theta)
        probs  = torch.sigmoid(logits)

        if loss_kind == "beta":
            mu = probs.clamp(min=eps, max=1.0 - eps)
            loss = beta_nll(asked_ys, mu, phi_t).mean()
        elif loss_kind == "binary":
            loss = -Bernoulli(probs=probs).log_prob(asked_ys).mean()
        else:
            raise ValueError(f"Unknown loss_kind: {loss_kind}")
        
        loss.backward()
        return loss
    
    theta = trainer([theta], optim, closure)[0]
    return theta.detach()

def _select_next_1pl(theta, rem_discri, rem_z):
    return torch.argmin(torch.abs(theta + rem_z)).item()

def _select_next_2pl(theta, rem_discri, rem_z):
    fi = _compute_fisher_info_2pl(theta, rem_discri, rem_z)
    return torch.argmax(fi).item()

def _compute_fisher_info_2pl(theta, rem_discri, rem_z):
    p = torch.sigmoid(rem_discri * (theta - rem_z))
    return p * (1 - p)

def _cat_core(ys, zs, device, estimator_fn, select_next_fn, discris=None, budget=50, init_frac=0.2):
    rem_y, rem_z = ys.clone(), zs.clone()
    rem_discri = discris.clone() if discris is not None else None

    # phase 1: pick spanning items and estimate theta_init once
    init_idx = _span_idxs(rem_z, rem_y, int(init_frac * budget))
    asked_y, asked_z = rem_y[init_idx], rem_z[init_idx]
    asked_discri = rem_discri[init_idx] if rem_discri is not None else None

    mask = torch.ones(rem_y.shape[0], dtype=torch.bool, device=device)
    mask[torch.tensor(init_idx, device=device, dtype=torch.long)] = False
    rem_y, rem_z = rem_y[mask], rem_z[mask]
    rem_discri = rem_discri[mask] if discris is not None else None

    theta = torch.zeros(1, device=device)
    theta = estimator_fn(theta, asked_y, asked_discri, asked_z, device)
    thetas = [theta.clone().item()]
    
    # phase 2: Fisher-info CAT for remaining budget
    asked = asked_y.numel()
    while asked < budget and rem_y.numel() > 0:
        i = select_next_fn(theta, rem_discri, rem_z)
        y_i, z_i = rem_y[i], rem_z[i]
        discri_i = rem_discri[i] if rem_discri is not None else None

        rem_y = torch.cat([rem_y[:i], rem_y[i+1:]])
        rem_z = torch.cat([rem_z[:i], rem_z[i+1:]])
        rem_discri = torch.cat([rem_discri[:i], rem_discri[i+1:]]) if rem_discri is not None else None

        if torch.isnan(y_i):
            continue

        asked_y = torch.cat([asked_y, y_i.view(1)])
        asked_z = torch.cat([asked_z, z_i.view(1)])
        asked_discri = torch.cat([asked_discri, discri_i.view(1)]) if rem_discri is not None else None

        theta = estimator_fn(theta, asked_y, asked_discri, asked_z, device)
        thetas.append(theta.clone().item())
        asked += 1

    return thetas

def _span_idxs(zs, ys, k, trim=0.10):
    lo, hi = torch.quantile(zs, torch.tensor([trim, 1 - trim], device=zs.device))
    targets = torch.linspace(lo, hi, steps=k, device=zs.device)
    idxs = []
    used = torch.zeros_like(zs, dtype=torch.bool)
    for t in targets:
        d = torch.abs(zs - t)
        d[used | torch.isnan(ys)] = float("inf")
        i = torch.argmin(d).item()
        idxs.append(i)
        used[i] = True
    return idxs





def trainer(parameters, optim, closure, n_iter=100, verbose=False, eps=1e-6):
    pbar = tqdm(range(n_iter)) if verbose else range(n_iter)
    for iteration in pbar:
        if iteration > 0:
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(previous_parameters, parameters)
            )
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            if verbose:
                pbar.set_postfix({"grad_norm": grad_norm, "d_parameter": d_parameters, "d_loss": d_loss})
            if d_loss < eps and d_parameters < eps and grad_norm < eps:
                break
            
    return parameters

def beta_nll(y, mu, phi):
    a = mu * phi
    b = (1.0 - mu) * phi
    return -((a - 1) * torch.log(y) + (b - 1) * torch.log1p(-y) - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)))






def compute_pass_iatk_gt(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_pass_datk_gts(data2d: np.ndarray) -> np.ndarray:
    assert isinstance(data2d, np.ndarray)
    n_items, n_samples = data2d.shape
    k_range = np.arange(1, n_samples + 1)
    per_item = []
    for i in range(n_items):
        arr = data2d[i]
        valid = ~np.isnan(arr) # if data2d is torch.tensor, this line return a list of 255
        n = int(valid.sum())
        c = int(np.nansum(arr))
        per_item.append([
            compute_pass_iatk_gt(n, c, k)
            for k in k_range
        ])
    return np.nanmean(np.vstack(per_item), axis=0)

def compute_pass_iatk_irt(pass_iat1: float, k: int) -> float:
    return 1.0 - (1.0 - pass_iat1) ** k

def compute_pass_datk_irt(irt_probs: np.ndarray, n_samples) -> np.ndarray:
    n_items = irt_probs.shape[0]
    k_range = np.arange(1, n_samples + 1)
    per_item = []
    for i in range(n_items):
        per_item.append([
            compute_pass_iatk_irt(irt_probs[i], k)
            for k in k_range
        ])
    return np.nanmean(np.vstack(per_item), axis=0)




 
def visualize_response_matrix(results, value, filename):
    # Extract the groups labels in the order of the columns
    group_values = results.columns.get_level_values("scenario")

    # Identify the boundaries where the group changes
    boundaries = []
    for i in range(1, len(group_values)):
        if group_values[i] != group_values[i - 1]:
            boundaries.append(i - 0.5)  # using 0.5 to place the line between columns

    # Visualize the results with a matrix: red is 0, white is -1 and blue is 1
    cmap = mcolors.ListedColormap(["white", "red", "blue"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Calculate midpoints for each group label
    groups_list = list(group_values)
    group_names = []
    group_midpoints = []
    current_group = groups_list[0]
    start_index = 0
    for i, grp in enumerate(groups_list):
        if grp != current_group:
            midpoint = (start_index + i - 1) / 2.0
            group_names.append(current_group)
            group_midpoints.append(midpoint)
            current_group = grp
            start_index = i
    # Add the last group
    midpoint = (start_index + len(groups_list) - 1) / 2.0
    group_names.append(current_group)
    group_midpoints.append(midpoint)

    # Define the minimum spacing between labels (e.g., 100 units)
    min_spacing = 100
    last_label_pos = -float("inf")
    # Plot the matrix
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, ax = plt.subplots(figsize=(20, 10))
        cax = ax.matshow(value, aspect="auto", cmap=cmap, norm=norm)

        # Add vertical lines at each boundary
        for b in boundaries:
            ax.axvline(x=b, color="black", linewidth=0.25, linestyle="--", alpha=0.5)
        
        # Add group labels above the matrix, only if they're spaced enough apart
        for name, pos in zip(group_names, group_midpoints):
            if pos - last_label_pos >= min_spacing:
                ax.text(pos, -5, name, ha='center', va='bottom', rotation=90, fontsize=3)
                last_label_pos = pos

        # Add model labels on the y-axis
        ax.set_yticks(range(len(results.index)))
        ax.set_yticklabels(results.index, fontsize=3)

        # Add a colorbar
        cbar = plt.colorbar(cax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["-1", "0", "1"])
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()
