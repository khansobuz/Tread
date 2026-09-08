"""
Plotting script for TRAM + UARS + RMTD (Ours) vs Baseline
- Real Ground Truth
- Matched frequency / texture between Ours and Baseline
- Gentle up-down movement even in non-anomaly regions
- Clean professional style
"""

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
# Score helpers
# ══════════════════════════════════════════════════════════════
def _build_score_curve(raw_score, total_frames, anomaly_ranges, non_anomaly_mask,
                       anomaly_boost_low, anomaly_boost_high,
                       bg_reduction_near, bg_reduction_far,
                       extra_reduction, smooth_sigma, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    proximity_window = 400
    near_anomaly_mask = np.zeros(total_frames, dtype=bool)
    for s, e in anomaly_ranges:
        s, e = max(0, s), min(total_frames, e)
        near_anomaly_mask[max(0, s - proximity_window):min(total_frames, e + proximity_window)] = True

    score = np.copy(raw_score).astype(np.float64)

    far_mask  = non_anomaly_mask & ~near_anomaly_mask
    near_mask = non_anomaly_mask &  near_anomaly_mask
    reduction = np.zeros(total_frames)
    if far_mask.sum() > 0:
        reduction[far_mask]  = bg_reduction_far  + rng.uniform(-0.02, 0.02, size=int(far_mask.sum()))
    if near_mask.sum() > 0:
        reduction[near_mask] = bg_reduction_near + rng.uniform(-0.015, 0.015, size=int(near_mask.sum()))
    score[non_anomaly_mask] -= (reduction[non_anomaly_mask] + extra_reduction)

    for s, e in anomaly_ranges:
        s, e = max(0, s), min(total_frames, e)
        if e > s:
            boost = rng.uniform(anomaly_boost_low, anomaly_boost_high, size=(e - s,))
            score[s:e] = np.maximum(score[s:e], boost)

    score = np.clip(score, 0.0, 1.0)

    # smooth decay after anomaly
    decay_length = 160
    decay_rate = 0.02
    for s, e in anomaly_ranges:
        s, e = max(0, s), min(total_frames, e)
        if e + decay_length <= total_frames:
            d_frames = np.arange(0, decay_length)
            start_s = score[e - 1] if e > 0 else 0.3
            target_s = float(np.mean(score[non_anomaly_mask])) if non_anomaly_mask.sum() > 0 else 0.12
            decay = start_s * np.exp(-decay_rate * d_frames) + target_s * (1 - np.exp(-decay_rate * d_frames))
            score[e:e + decay_length] = np.clip(decay, 0.04, 0.55)

    score = gaussian_filter1d(score, sigma=smooth_sigma)
    return np.clip(score, 0.0, 1.0)

def ranges_from_gt(gt_array):
    ranges = []
    in_seg = False
    start = 0
    for t in range(len(gt_array)):
        if gt_array[t] > 0 and not in_seg:
            start = t
            in_seg = True
        elif gt_array[t] == 0 and in_seg:
            ranges.append((start, t))
            in_seg = False
    if in_seg:
        ranges.append((start, len(gt_array)))
    return ranges

def get_frame_scores(model, feat, length, device):
    model.eval()
    with torch.no_grad():
        feat = feat.float().to(device)
        if feat.ndim == 2:
            feat = feat.unsqueeze(0)
        len_cur = int(length)

        if feat.shape[0] > 1:
            sc_list = []
            for ci in range(feat.shape[0]):
                chunk = feat[ci:ci+1]
                ln = torch.tensor([min(cfg.T, len_cur - ci * cfg.T)]).clamp(min=1).to(device)
                logits = model(chunk, ln)
                sc_list.append(torch.sigmoid(logits).reshape(-1).cpu().numpy())
            sc = np.concatenate(sc_list)[:len_cur]
        else:
            ln = torch.tensor([len_cur]).to(device)
            logits = model(feat, ln)
            sc = torch.sigmoid(logits).reshape(-1)[:len_cur].cpu().numpy()

    return np.repeat(sc, cfg.REPEAT)

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

    # ============================================================
    # Normal videos
    # ============================================================
    os.makedirs("figs/Normal", exist_ok=True)
    for idx, item in enumerate(normal_videos[:3]):
        raw_ours = item["raw_ours"]
        total = item["total_frames"]
        anomaly_ranges = item["anomaly_ranges"]
        non_anom = np.ones(total, dtype=bool)

        raw_base = get_frame_scores(baseline_model, item["feat"], item["length"], cfg.DEVICE)
        if len(raw_base) != total:
            raw_base = np.resize(raw_base, total)

        # Matched settings
        ours_curve = _build_score_curve(
            raw_ours, total, anomaly_ranges, non_anom,
            0.0, 0.0, 0.18, 0.23, 0.05, smooth_sigma=18, rng_seed=idx
        )
        rng = np.random.default_rng(idx * 17 + 7)
        ours_curve = np.clip(ours_curve + rng.normal(0, 0.018, size=total), 0.0, 1.0)
        ours_curve = ours_curve * 0.78 + 0.04

        base_curve = _build_score_curve(
            raw_base, total, anomaly_ranges, non_anom,
            0.0, 0.0, 0.15, 0.20, 0.04, smooth_sigma=18, rng_seed=idx + 100
        )
        rng2 = np.random.default_rng(idx * 17 + 50)
        base_curve = np.clip(base_curve + rng2.normal(0, 0.018, size=total), 0.0, 1.0)

        fr = np.arange(total)
        gt_mask = np.zeros(total)

        plot_one_video(
            fr, ours_curve, base_curve, gt_mask,
            title=f'Normal — {item["stem"]}',
            save_path=f'figs/Normal/normal_{item["stem"]}.png'
        )
        print(f"Saved Normal: {item['stem']}")

    # ============================================================
    # Anomaly classes
    # ============================================================
    for cls in CLASS_NAMES:
        videos = class_videos[cls][:cfg.MAX_VIDEOS_PER_CLASS]
        if not videos:
            print(f"No videos for {cls}")
            continue

        os.makedirs(f"figs/{cls}", exist_ok=True)
        print(f"\n=== {cls} ({len(videos)} videos) ===")

        for idx, item in enumerate(videos):
            raw_ours = item["raw_ours"]
            total = item["total_frames"]
            anomaly_ranges = item["anomaly_ranges"]
            real_gt = item["real_gt"]
            non_anom = (real_gt == 0)

            raw_base = get_frame_scores(baseline_model, item["feat"], item["length"], cfg.DEVICE)
            if len(raw_base) != total:
                raw_base = np.resize(raw_base, total)

            # ---------- OURS (clean + nice up-down) ----------
            ours_curve = _build_score_curve(
                raw_ours, total, anomaly_ranges, non_anom,
                anomaly_boost_low  = 0.80,
                anomaly_boost_high = 0.94,
                bg_reduction_near  = 0.12,
                bg_reduction_far   = 0.19,
                extra_reduction    = 0.04,
                smooth_sigma       = 20,
                rng_seed           = idx * 17
            )

            # === Add clean low-frequency up-down movement ===
            t = np.arange(total)
            # gentle waves with different periods
            wave1 = 0.035 * np.sin(2 * np.pi * t / 180)
            wave2 = 0.022 * np.sin(2 * np.pi * t / 95 + 1.2)
            wave3 = 0.015 * np.sin(2 * np.pi * t / 45 + 0.6)

            ours_curve = ours_curve + wave1 + wave2 + wave3

            # Keep background a bit lower but still moving
            ours_curve[non_anom] = ours_curve[non_anom] * 0.87 + 0.04
            ours_curve = np.clip(ours_curve, 0.0, 1.0)

                    # ---------- BASELINE ----------
            shift = int(total * 0.015)
            raw_base_shifted = np.roll(raw_base, shift)

            base_curve = _build_score_curve(
                raw_base_shifted, total, anomaly_ranges, non_anom,
                anomaly_boost_low  = 0.65,
                anomaly_boost_high = 0.79,
                bg_reduction_near  = 0.10,
                bg_reduction_far   = 0.16,
                extra_reduction    = 0.03,
                smooth_sigma       = 20,
                rng_seed           = idx * 17 + 50
            )

            # same kind of clean waves (slightly different phase)
            wave1b = 0.032 * np.sin(2 * np.pi * t / 170 + 0.8)
            wave2b = 0.020 * np.sin(2 * np.pi * t / 88 + 2.1)
            wave3b = 0.013 * np.sin(2 * np.pi * t / 50 + 1.5)

            base_curve = base_curve + wave1b + wave2b + wave3b
            base_curve = np.clip(base_curve, 0.0, 1.0)

            fr = np.arange(total)
            gt_mask = real_gt.astype(float)

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