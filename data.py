import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import cfg

# ══════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════
def get_tram_name(path_or_name):
    name = os.path.basename(str(path_or_name))
    name = name.replace(".npy", "").replace(".mp4", "").replace(".pt", "")
    if "__" in name:
        name = name.split("__")[0]
    return name + ".pt"

def fix_path(p):
    if os.path.exists(p):
        return p
    return os.path.join(cfg.FEATURE_ROOT,
                        os.path.basename(os.path.dirname(p)),
                        os.path.basename(p))

def get_prompt_text():
    return list(cfg.LABEL_MAP.values())

def uniform_extract(feat, t_max):
    new_feat = np.zeros((t_max, feat.shape[1]), np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    for i in range(t_max):
        if r[i] != r[i + 1]:
            new_feat[i] = np.mean(feat[r[i]:r[i + 1]], 0)
        else:
            new_feat[i] = feat[min(r[i], len(feat) - 1)]
    return new_feat

def pad_feat(feat, min_len):
    if feat.shape[0] <= min_len:
        return np.pad(feat, ((0, min_len - feat.shape[0]), (0, 0)), mode='constant')
    return feat

def process_feat(feat, length):
    if feat.shape[0] > length:
        return uniform_extract(feat, length), length
    return pad_feat(feat, length), feat.shape[0]

def process_split(feat, length):
    n = feat.shape[0]
    if n < length:
        return pad_feat(feat, length)[np.newaxis], n
    chunks = []
    for i in range(int(n / length) + 1):
        chunks.append(pad_feat(feat[i * length:(i + 1) * length], length)[np.newaxis])
    return np.concatenate(chunks, 0), n

def align_tram(p, r, y, w, target_len):
    """Align TRAM outputs (p, r, y, w) to fixed temporal length T."""
    N = p.shape[0]
    if N == target_len:
        return p, r, y, w
    if N > target_len:
        edges = np.linspace(0, N, target_len + 1).astype(int)
        p_new, r_new, y_new, w_new = [], [], [], []
        for i in range(target_len):
            s, e = edges[i], max(edges[i + 1], edges[i] + 1)
            p_new.append(p[s:e].mean())
            mid = (s + e - 1) // 2
            r_new.append(r[mid]); y_new.append(y[mid]); w_new.append(w[mid])
        return torch.stack(p_new), torch.stack(r_new), torch.stack(y_new), torch.stack(w_new)
    pad = target_len - N
    return (F.pad(p, (0, pad)),
            F.pad(r, (0, 0, 0, pad)),
            F.pad(y, (0, 0, 0, pad)),
            F.pad(w, (0, 0, 0, pad)))

# ══════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════
class UCFTrainDataset(Dataset):
    def __init__(self, csv_path, tram_dir, normal=True):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
        df["label"] = df["label"].str.strip()
        df["fpath"] = df["path"].apply(fix_path)
        if normal:
            df = df[df["label"] == "Normal"].reset_index(drop=True)
        else:
            df = df[df["label"] != "Normal"].reset_index(drop=True)
        valid = []
        for _, row in df.iterrows():
            if os.path.exists(os.path.join(tram_dir, get_tram_name(row.get("path", row["fpath"])))):
                valid.append(row)
        self.df = pd.DataFrame(valid).reset_index(drop=True)
        self.tram_dir = tram_dir
        print(f"  {'Normal' if normal else 'Anomaly'} with TRAM: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            feat = np.load(row["fpath"]).astype(np.float32)
            if feat.ndim == 1:
                feat = feat[np.newaxis]
        except:
            feat = np.zeros((1, cfg.FEAT_DIM), np.float32)
        feat, length = process_feat(feat, cfg.T)
        label = row["label"]
        is_anomaly = 0.0 if label == "Normal" else 1.0
        cat_idx = cfg.CATEGORIES.index(label) if label in cfg.CATEGORIES else 0
        # Load pre-computed TRAM relations: p (MLLM conf), r (relative), y (ordering), w (UARS weight)
        data = torch.load(os.path.join(self.tram_dir, get_tram_name(row.get("path", row["fpath"]))), map_location="cpu")
        p, r, y, w = align_tram(data["p"], data["r"], data["y"], data["w"], cfg.T)
        return (torch.tensor(feat), torch.tensor(length),
                torch.tensor(is_anomaly), torch.tensor(cat_idx),
                p, r, y, w)

class UCFTestDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
        df["label"] = df["label"].str.strip()
        df["fpath"] = df["path"].apply(fix_path)
        self.df = df
        print(f"  Test clips: {len(df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            feat = np.load(row["fpath"]).astype(np.float32)
            if feat.ndim == 1:
                feat = feat[np.newaxis]
        except:
            feat = np.zeros((1, cfg.FEAT_DIM), np.float32)
        feat, length = process_split(feat, cfg.T)
        return torch.tensor(feat), row["label"], torch.tensor(length)