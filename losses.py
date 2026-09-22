import torch
import torch.nn.functional as F
from config import cfg

# ══════════════════════════════════════════════════════════════
# LOSSES  (aligned with paper: TRAM / UARS / RMTD)
# ══════════════════════════════════════════════════════════════

# ── UARS: Uncertainty-Aware Relation Selection ────────────────
# Paper: soft-threshold reliability weight w_ij from ρ_ij
#   w = 1 if ρ ≥ τ_h,  linear ramp if τ_l < ρ < τ_h,  0 if ρ ≤ τ_l
def apply_uars(w, tau_h=cfg.TAU_H, tau_l=cfg.TAU_L):
    """Apply UARS soft-thresholding on pre-computed reliability weights."""
    new_w = w.clone()
    new_w[w >= tau_h] = 1.0
    mask = (w > tau_l) & (w < tau_h)
    new_w[mask] = (w[mask] - tau_l) / (tau_h - tau_l + 1e-8)
    new_w[w <= tau_l] = 0.0
    return new_w

# ── Base WSVAD losses (L_WS) ──────────────────────────────────
def clas2_loss(logits1, labels_bin, lengths, device):
    """Binary instance-level classification loss."""
    B = logits1.shape[0]
    probs = torch.sigmoid(logits1).reshape(B, -1)
    inst = []
    for i in range(B):
        L = max(1, min(int(lengths[i]), probs.shape[1]))
        k = max(1, min(L // 16 + 1, L))
        inst.append(probs[i, :L].topk(k).values.mean().view(1))
    inst = torch.cat(inst)
    return F.binary_cross_entropy(inst, labels_bin.float().to(device))

def clasm_loss(logits2, labels_cls, lengths, device):
    """Multi-class prompt alignment loss."""
    B = logits2.shape[0]
    inst = []
    lbl = (labels_cls / (labels_cls.sum(1, keepdim=True) + 1e-8)).to(device)
    for i in range(B):
        L = max(1, min(int(lengths[i]), logits2.shape[1]))
        k = max(1, min(L // 16 + 1, L))
        tmp, _ = logits2[i, :L].topk(k, dim=0, largest=True)
        inst.append(tmp.mean(0, keepdim=True))
    inst = torch.cat(inst, dim=0)
    return -torch.mean(torch.sum(lbl * F.log_softmax(inst, dim=1), dim=1))

# ── Soft TRAM loss (MSE on absolute MLLM confidence p) ────────
def soft_tram_loss(scores, p_teacher, lengths):
    """Soft distillation of TRAM absolute confidence p_i → student scores."""
    B = scores.shape[0]
    loss = 0.0
    for i in range(B):
        L = max(1, min(int(lengths[i]), scores.shape[1]))
        loss = loss + F.mse_loss(scores[i, :L], p_teacher[i, :L])
    return loss / B

# ── RMTD: Relative-Margin Temporal Distillation ───────────────
# Paper Eq:
#   r̂_ij = S_θ(c_i) - S_θ(c_j)
#   m_ij  = m_min + (m_max - m_min) * |r_ij| / M
#   ℓ_ij  = [m_ij - y_ij * r̂_ij]_+
#   L_RMTD = Σ w_ij ℓ_ij  /  Σ w_ij
def rmtd_loss(scores, r, y, w):
    """
    Relative-Margin Temporal Distillation.
    scores : student anomaly scores S_θ  [B, T]
    r      : teacher relative responses r_ij  [B, T, 2K]
    y      : relational states y_ij ∈ {-1,0,+1}  [B, T, 2K]
    w      : UARS reliability weights w_ij  [B, T, 2K]
    """
    B, T = scores.shape
    K2 = r.shape[-1]
    loss = 0.0
    count = 0.0
    for b in range(B):
        s = scores[b]
        max_r = r[b].abs().max() + 1e-8          # M = max |r_ab| + ε
        for i in range(0, T, 2):
            for k in range(K2):
                if w[b, i, k] <= 0:
                    continue
                # map k → temporal offset (past / future neighbors)
                offset = -(cfg.K - k) if k < cfg.K else (k - cfg.K) + 1
                j = i + offset
                if j < 0 or j >= T:
                    continue
                r_ij = r[b, i, k]
                y_ij = y[b, i, k].float()
                w_ij = w[b, i, k]
                # adaptive margin
                m_ij = cfg.M_MIN + (cfg.M_MAX - cfg.M_MIN) * (r_ij.abs() / max_r)
                # hinge: [m_ij - y_ij * (s_i - s_j)]_+
                loss = loss + w_ij * F.relu(m_ij - y_ij * (s[i] - s[j]))
                count = count + w_ij
    if count < 1e-6:
        return torch.tensor(0.0, device=scores.device)
    return loss / count

def get_label_vectors(cat_idxs):
    C = len(cfg.CATEGORIES)
    vecs = torch.zeros(len(cat_idxs), C)
    for i, c in enumerate(cat_idxs):
        vecs[i, int(c)] = 1.0
    return vecs