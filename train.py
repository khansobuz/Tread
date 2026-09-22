import os
import random
import warnings
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

from config import cfg
from data import get_prompt_text, UCFTrainDataset, UCFTestDataset
from model import VADModel
from losses import (
    apply_uars, clas2_loss, clasm_loss,
    soft_tram_loss, rmtd_loss, get_label_vectors
)

warnings.filterwarnings("ignore")
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)

# ══════════════════════════════════════════════════════════════
# EVAL
# ══════════════════════════════════════════════════════════════
def evaluate(epoch, model, test_loader, device, best_auc, tag=""):
    model.eval()
    prompt_text = get_prompt_text()
    gt = np.load(cfg.GT_NPY)
    gt_bin = (gt > 0).astype(int)
    all_scores = []
    with torch.no_grad():
        for feat, label, length in test_loader:
            feat = feat.squeeze(0).float().to(device)
            len_cur = int(length[0])
            if feat.ndim == 2:
                feat = feat.unsqueeze(0)
            sc_list = []
            for ci in range(feat.shape[0]):
                chunk = feat[ci:ci+1]
                ln = torch.tensor([min(cfg.T, max(1, len_cur - ci * cfg.T))])
                scores, logits1, logits2, _ = model(chunk, ln, prompt_text)
                sc1 = scores.reshape(-1).cpu().numpy()
                sc2 = (1 - logits2.softmax(-1)[..., 0]).reshape(-1).cpu().numpy()
                sc = 0.7 * sc1 + 0.3 * sc2
                sc_list.append(sc)
            sc = np.concatenate(sc_list)[:len_cur]
            if len(sc) > 5:
                sc = np.convolve(sc, np.ones(5)/5.0, mode='same')
            all_scores.append(sc)
    pred = np.repeat(np.concatenate(all_scores), cfg.REPEAT)
    def match(arr, n):
        return arr[:n] if len(arr) > n else np.pad(arr, (0, n - len(arr)), mode='edge')
    pred = match(pred, len(gt_bin))
    auc = roc_auc_score(gt_bin, pred)
    marker = " *** NEW BEST ***" if auc > best_auc else ""
    prefix = f"[Test]{tag}" if tag else "[Test]"
    print(f"{prefix} Ep {epoch:02d} | AUC:{auc:.4f}  Best:{max(best_auc, auc):.4f}{marker}")
    if auc > best_auc:
        os.makedirs("checkpoint", exist_ok=True)
        torch.save({"net": model.state_dict(), "auc": auc}, "checkpoint/best_vadclip_tram.pth")
    return auc

# ══════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════
def train_epoch(epoch, model, normal_loader, anomaly_loader, optimizer, device, test_loader, best_auc):
    model.train()
    prompt_text = get_prompt_text()
    total = {"c2": 0.0, "cm": 0.0, "soft": 0.0, "rmtd": 0.0}
    steps = 0
    use_rmtd = epoch <= cfg.RMTD_END_EP
    n_iter = iter(normal_loader)
    a_iter = iter(anomaly_loader)
    n_steps = min(len(normal_loader), len(anomaly_loader))
    for _ in range(n_steps):
        nd = next(n_iter); ad = next(a_iter)
        def pack(d):
            feat, length, label, cat, p, r, y, w = d
            return (feat.float().to(device), length.to(device), label.float().to(device),
                    cat.to(device), p.to(device), r.to(device), y.to(device), w.to(device))
        nf, nlen, nlab, ncat, np_, nr, ny, nw = pack(nd)
        af, alen, alab, acat, ap, ar, ay, aw = pack(ad)
        visual  = torch.cat([nf, af], 0)
        lengths = torch.cat([nlen, alen], 0)
        labels  = torch.cat([nlab, alab], 0)
        cats    = torch.cat([ncat, acat], 0)
        p       = torch.cat([np_, ap], 0)
        r       = torch.cat([nr, ar], 0)
        y       = torch.cat([ny, ay], 0)
        w       = apply_uars(torch.cat([nw, aw], 0))   # UARS soft-threshold
        scores, logits1, logits2, _ = model(visual, lengths, prompt_text)
        labels_cls = get_label_vectors(cats.cpu()).to(device)
        # L_WS = clas2 + clasm
        l_c2   = clas2_loss(logits1, labels, lengths, device)
        l_cm   = clasm_loss(logits2, labels_cls, lengths, device)
        # Soft TRAM (MSE on p)
        l_soft = soft_tram_loss(scores, p, lengths)
        # RMTD (relative-margin distillation)
        l_rmtd = rmtd_loss(scores, r, y, w) if use_rmtd else torch.tensor(0.0, device=device)
        # Total: L = L_WS + λ_soft * L_soft + λ_R * L_RMTD
        loss = l_c2 + l_cm + cfg.LAMBDA_SOFT * l_soft + (cfg.LAMBDA_R if use_rmtd else 0.0) * l_rmtd
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total["c2"] += l_c2.item()
        total["cm"] += l_cm.item()
        total["soft"] += l_soft.item()
        total["rmtd"] += l_rmtd.item()
        steps += 1
        if steps % cfg.EVAL_EVERY == 0:
            print(f"  ep:{epoch} step:{steps}  C2:{total['c2']/steps:.4f}  "
                  f"SOFT:{total['soft']/steps:.4f}  RMTD:{total['rmtd']/steps:.4f}  "
                  f"(rmtd={'ON' if use_rmtd else 'OFF'})")
            auc = evaluate(epoch, model, test_loader, device, best_auc, tag=f" step:{steps}")
            if auc > best_auc:
                best_auc = auc
            model.train()
    print(f"[Train] Ep {epoch:02d} | C2:{total['c2']/max(steps,1):.4f} CM:{total['cm']/max(steps,1):.4f} "
          f"SOFT:{total['soft']/max(steps,1):.4f} RMTD:{total['rmtd']/max(steps,1):.4f} | rmtd={'ON' if use_rmtd else 'OFF'}")
    return best_auc

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print(f"Device: {cfg.DEVICE}")
    print("VadCLIP + TRAM soft + light RMTD | step-wise + epoch eval | target 88%+")
    print("Loading datasets...")
    normal_ds  = UCFTrainDataset(cfg.TRAIN_CSV, cfg.TRAM_TRAIN_DIR, normal=True)
    anomaly_ds = UCFTrainDataset(cfg.TRAIN_CSV, cfg.TRAM_TRAIN_DIR, normal=False)
    test_ds    = UCFTestDataset(cfg.TEST_CSV)
    print(f"Normal: {len(normal_ds)} | Anomaly: {len(anomaly_ds)}")
    if len(normal_ds) == 0 or len(anomaly_ds) == 0:
        print("ERROR: no TRAM files matched")
        return
    normal_loader  = DataLoader(normal_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    anomaly_loader = DataLoader(anomaly_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    test_loader    = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    model = VADModel().to(cfg.DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    ckpt_path = "checkpoint/best_vadclip_tram.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["net"], strict=False)
        print(f"Loaded previous best checkpoint | AUC={ckpt.get('auc', 'NA')}")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    best_auc, no_improve = 0.0, 0
    for epoch in range(1, cfg.EPOCHS + 1):
        best_auc = train_epoch(epoch, model, normal_loader, anomaly_loader,
                               optimizer, cfg.DEVICE, test_loader, best_auc)
        auc = evaluate(epoch, model, test_loader, cfg.DEVICE, best_auc, tag=" EPOCH")
        if auc > best_auc:
            best_auc = auc
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step()
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e} | no_improve: {no_improve}\n")
        if no_improve >= cfg.PATIENCE:
            print(f"Early stop at epoch {epoch}. Best AUC: {best_auc:.4f}")
            break
    print(f"\nFinal Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()