
import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import Dict

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
class Cfg:
    FEATURE_ROOT  = r"C:\Users\khanm\Desktop\lab_project\Open_vocab\UCFClipFeatures"
    EXTRA_DIR     = r"C:\Users\khanm\Desktop\lab_project\Open_vocab\ucf_extra"
    TEST_CSV      = os.path.join(EXTRA_DIR, "ucf_CLIP_rgbtest.csv")
    GT_NPY        = os.path.join(EXTRA_DIR, "gt_ucf.npy")

    FEAT_DIM      = 512
    T             = 256
    REPEAT        = 16

    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED          = 42

    CKPT_PATH     = "checkpoint/best_student.pth"
    MAX_VIDEOS_PER_CLASS = 8

cfg = Cfg()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)

CLASS_NAMES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
    'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
    'Stealing', 'Vandalism'
]

# ══════════════════════════════════════════════════════════════
# FEATURE HELPERS
# ══════════════════════════════════════════════════════════════
def pad_feat(feat, min_len):
    if feat.shape[0] <= min_len:
        return np.pad(feat, ((0, min_len - feat.shape[0]), (0, 0)), mode='constant', constant_values=0)
    return feat

def process_split(feat, length):
    n = feat.shape[0]
    if n < length:
        return pad_feat(feat, length)[np.newaxis], n
    chunks = []
    for i in range(int(n / length) + 1):
        chunks.append(pad_feat(feat[i * length:(i + 1) * length], length)[np.newaxis])
    return np.concatenate(chunks, 0), n

def fix_path(p):
    return os.path.join(cfg.FEATURE_ROOT, os.path.basename(os.path.dirname(p)), os.path.basename(p))

# ══════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════
class LightweightStudent(nn.Module):
    def __init__(self, feat_dim=512, d_model=512, n_head=8, n_layers=3, ff_dim=1024,
                 dropout=0.15, max_len=256):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, lengths):
        B, T, _ = x.shape
        h = self.input_proj(x)
        pos = self.pos_embed(torch.arange(T, device=x.device)).unsqueeze(0)
        h = h + pos
        key_padding_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for b in range(B):
            L = int(lengths[b])
            if L < T:
                key_padding_mask[b, L:] = True
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return self.head(h).squeeze(-1)


class BaselineStudent(nn.Module):
    def __init__(self, feat_dim=512, d_model=256, n_head=4, n_layers=2, ff_dim=512,
                 dropout=0.1, max_len=256):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        h = self.input_proj(x)
        pos = self.pos_embed(torch.arange(T, device=x.device)).unsqueeze(0)
        h = h + pos
        key_padding_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for b in range(B):
            L = int(lengths[b])
            if L < T:
                key_padding_mask[b, L:] = True
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.head(h).squeeze(-1)

# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════
class UCFTestPlotDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
        df["label"] = df["label"].str.strip()
        df["fpath"] = df["path"].apply(fix_path)
        df["stem"]  = df["path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
        self.df = df
        print(f"Test clips loaded: {len(df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            feat = np.load(row["fpath"]).astype(np.float32)
            if feat.ndim == 1:
                feat = feat[np.newaxis]
        except Exception:
            feat = np.zeros((1, cfg.FEAT_DIM), np.float32)
        feat, length = process_split(feat, cfg.T)
        return {
            "feat": torch.tensor(feat),
            "label": row["label"],
            "length": torch.tensor(length),
            "stem": row["stem"],
            "path": row["path"],
        }



# ══════════════════════════════════════════════════════════════
# Clean plotting function
# ══════════════════════════════════════════════════════════════
def plot_one_video(fr, ours_curve, base_curve, gt_mask, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 3.5))

    # Ground Truth - medium pink
    ax.fill_between(fr, gt_mask, color='#ff99bb', alpha=0.48, label='Ground Truth', zorder=1)

    # Ours - clean blue
    ax.plot(fr, ours_curve,
            color='#0077bb',
            linewidth=1.7,
            linestyle='-',
            label='Ours',
            zorder=3)

    # Baseline - soft orange dashed
    ax.plot(fr, base_curve,
            color='#ee7733',
            linewidth=1.5,
            linestyle='--',
            label='Baseline',
            zorder=2)

    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, len(fr))

    ax.legend(loc='upper right', fontsize=10, framealpha=0.92)
    ax.grid(True, linestyle=':', alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("Device:", cfg.DEVICE)
    os.makedirs("figs", exist_ok=True)

    # Load models
    ours_model = LightweightStudent(
        feat_dim=512, d_model=512, n_head=8, n_layers=3,
        ff_dim=1024, dropout=0.15, max_len=256
    ).to(cfg.DEVICE)

    if os.path.isfile(cfg.CKPT_PATH):
        ckpt = torch.load(cfg.CKPT_PATH, map_location="cpu", weights_only=False)
        state = ckpt["net"] if "net" in ckpt else ckpt
        ours_model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {cfg.CKPT_PATH}  (AUC={ckpt.get('auc', 'N/A')})")
    else:
        print(f"[WARN] Checkpoint not found: {cfg.CKPT_PATH}")

    baseline_model = BaselineStudent().to(cfg.DEVICE)

    # Real GT
    gt_all = np.load(cfg.GT_NPY)
    print(f"Loaded real GT: shape = {gt_all.shape}")

    test_ds = UCFTestPlotDataset(cfg.TEST_CSV)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # Align GT
    print("\nAligning real GT with scores...")
    gt_ptr = 0
    all_items = []

    for sample in test_loader:
        label  = sample["label"][0]
        stem   = sample["stem"][0]
        feat   = sample["feat"].squeeze(0)
        length = sample["length"].item()

        raw_ours = get_frame_scores(ours_model, feat, length, cfg.DEVICE)
        total_frames = len(raw_ours)

        if gt_ptr + total_frames > len(gt_all):
            print(f"[WARN] GT overflow at {stem}")
            real_gt = np.zeros(total_frames)
        else:
            real_gt = gt_all[gt_ptr : gt_ptr + total_frames]
        gt_ptr += total_frames

        anomaly_ranges = ranges_from_gt(real_gt)

        all_items.append({
            "label": label,
            "stem": stem,
            "feat": feat,
            "length": length,
            "raw_ours": raw_ours,
            "real_gt": real_gt,
            "anomaly_ranges": anomaly_ranges,
            "total_frames": total_frames,
        })

    print(f"GT pointer finished at {gt_ptr} / {len(gt_all)}")

    class_videos = {c: [] for c in CLASS_NAMES}
    normal_videos = []
    for item in all_items:
        if item["label"] == "Normal":
            normal_videos.append(item)
        elif item["label"] in class_videos:
            class_videos[item["label"]].append(item)

  

            save_name = f'figs/{cls}/{cls.lower()}_{item["stem"]}.png'
            plot_one_video(
                fr, ours_curve, base_curve, gt_mask,
                title=f'{cls} — {item["stem"]}',
                save_path=save_name
            )
            print(f"  Saved: {save_name}")

    print("\n========== DONE ==========")
    print(f"GT used: {gt_ptr} frames out of {len(gt_all)}")
    print("Figures saved under figs/")
    print("==========================")

if __name__ == "__main__":
    main()
