

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import warnings
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
class Cfg:
    FEATURE_ROOT   = r"/home/sabuj_khan/research/projects/Open_vocab/UCFClipFeatures"
    EXTRA_DIR      = r"/home/sabuj_khan/research/projects/Open_vocab/ucf_extra"
    TRAIN_CSV      = os.path.join(EXTRA_DIR, "ucf_CLIP_rgb.csv")
    TEST_CSV       = os.path.join(EXTRA_DIR, "ucf_CLIP_rgbtest.csv")
    GT_NPY         = os.path.join(EXTRA_DIR, "gt_ucf.npy")
    TRAM_TRAIN_DIR = r"/home/sabuj_khan/research/projects/DMN/tram_outputs/train"

    FEAT_DIM       = 512
    T              = 256
    REPEAT         = 16

    EMBED_DIM      = 512
    VISUAL_WIDTH   = 512
    VISUAL_HEAD    = 1
    VISUAL_LAYERS  = 2
    ATTN_WINDOW    = 8

    K              = 2
    M_MIN          = 0.05
    M_MAX          = 0.35
    LAMBDA_SOFT    = 0.40
    LAMBDA_R       = 0.05
    RMTD_END_EP    = 12
    TAU_H          = 0.55
    TAU_L          = 0.20

    EPOCHS         = 25
    BATCH_SIZE     = 16
    LR             = 5e-5
    WEIGHT_DECAY   = 5e-4
    EVAL_EVERY     = 30          # step-wise eval
    PATIENCE       = 8
    SEED           = 1234
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

    CATEGORIES = [
        "Normal","Abuse","Arrest","Arson","Assault","Burglary",
        "Explosion","Fighting","RoadAccidents","Robbery",
        "Shooting","Shoplifting","Stealing","Vandalism"
    ]
    LABEL_MAP = {
        'Normal':'normal','Abuse':'abuse','Arrest':'arrest',
        'Arson':'arson','Assault':'assault','Burglary':'burglary',
        'Explosion':'explosion','Fighting':'fighting',
        'RoadAccidents':'roadAccidents','Robbery':'robbery',
        'Shooting':'shooting','Shoplifting':'shoplifting',
        'Stealing':'stealing','Vandalism':'vandalism'
    }

cfg = Cfg()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)

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
# LAYERS
# ══════════════════════════════════════════════════════════════
class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        orig = x.dtype
        return super().forward(x.float()).type(orig)

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, padding_mask=None):
        pm = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        am = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=pm, attn_mask=am)[0]

    def forward(self, x):
        x, pm = x
        x = x + self.attention(self.ln_1(x), pm)
        x = x + self.mlp(self.ln_2(x))
        return (x, pm)

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
        ])
    def forward(self, x):
        return self.resblocks(x)

class GraphConvolution(Module):
    def __init__(self, in_f, out_f, bias=False, residual=True):
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.weight = Parameter(torch.FloatTensor(in_f, out_f))
        self.bias = Parameter(torch.FloatTensor(out_f)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)
        if not residual:
            self.residual = lambda x: 0
        elif in_f == out_f:
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv1d(in_f, out_f, kernel_size=5, padding=2)

    def forward(self, inp, adj):
        out = adj.matmul(inp.matmul(self.weight))
        if self.bias is not None:
            out = out + self.bias
        if self.in_features != self.out_features and callable(self.residual):
            res = self.residual(inp.permute(0, 2, 1)).permute(0, 2, 1)
            out = out + res
        else:
            out = out + self.residual(inp)
        return out

class DistanceAdj(Module):
    def __init__(self):
        super().__init__()
        self.sigma = Parameter(torch.FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen):
        arith = np.arange(max_seqlen).reshape(-1, 1)
        dist = pdist(arith, metric='cityblock').astype(np.float32)
        dist = torch.from_numpy(squareform(dist)).to(cfg.DEVICE)
        dist = torch.exp(-dist / torch.exp(torch.tensor(1.)))
        return dist.unsqueeze(0).repeat(batch_size, 1, 1)

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

# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════
class VADModel(nn.Module):
    def __init__(self):
        super().__init__()
        W = cfg.VISUAL_WIDTH
        self.temporal = Transformer(
            width=W, layers=cfg.VISUAL_LAYERS, heads=cfg.VISUAL_HEAD,
            attn_mask=self._build_attn_mask(cfg.ATTN_WINDOW)
        )
        hw = W // 2
        self.gc1 = GraphConvolution(W, hw, residual=True)
        self.gc2 = GraphConvolution(hw, hw, residual=True)
        self.gc3 = GraphConvolution(W, hw, residual=True)
        self.gc4 = GraphConvolution(hw, hw, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(W, W)
        self.gelu = QuickGELU()
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(W, W * 4)), ("gelu", QuickGELU()), ("c_proj", nn.Linear(W * 4, W))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(W, W * 4)), ("gelu", QuickGELU()), ("c_proj", nn.Linear(W * 4, W))
        ]))
        self.classifier = nn.Linear(W, 1)
        self.pos_embed = nn.Embedding(cfg.T, W)
        nn.init.normal_(self.pos_embed.weight, std=0.01)

        import clip as clip_lib
        self.clipmodel, _ = clip_lib.load("ViT-B/16", cfg.DEVICE)
        for p in self.clipmodel.parameters():
            p.requires_grad = False
        self.dropout = nn.Dropout(0.15)

    def _build_attn_mask(self, attn_window):
        mask = torch.empty(cfg.T, cfg.T).fill_(float('-inf'))
        for i in range(int(cfg.T / attn_window)):
            s = i * attn_window
            e = min(s + attn_window, cfg.T)
            mask[s:e, s:e] = 0
        return mask

    def _adj4(self, x, lengths):
        x2 = x.matmul(x.permute(0, 2, 1))
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x2 = x2 / (x_norm.matmul(x_norm.permute(0, 2, 1)) + 1e-20)
        output = torch.zeros_like(x2)
        for i in range(len(lengths)):
            L = max(1, min(int(lengths[i]), x2.shape[1]))
            tmp = F.softmax(F.threshold(x2[i, :L, :L], 0.7, 0), dim=1)
            output[i, :L, :L] = tmp
        return output

    def encode_video(self, x, lengths):
        x = x.float()
        pos = self.pos_embed(torch.arange(cfg.T, device=x.device)).unsqueeze(0)
        x = self.dropout(x + pos)
        x = x.permute(1, 0, 2)
        x, _ = self.temporal((x, None))
        x = x.permute(1, 0, 2)
        adj = self._adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1 = self.gelu(self.gc2(self.gelu(self.gc1(x, adj)), adj))
        x2 = self.gelu(self.gc4(self.gelu(self.gc3(x, disadj)), disadj))
        x = self.linear(torch.cat([x1, x2], 2))
        return self.dropout(x)

    def encode_text(self, text):
        import clip as clip_lib
        with torch.no_grad():
            tokens = clip_lib.tokenize(text).to(cfg.DEVICE)
            feats = F.normalize(self.clipmodel.encode_text(tokens).float(), dim=-1)
        return feats

    def forward(self, visual, lengths, prompt_text):
        vf = self.encode_video(visual, lengths)
        logits1 = self.classifier(vf + self.mlp2(vf))

        text_feat = self.encode_text(prompt_text)
        logits_attn = logits1.permute(0, 2, 1)
        v_attn = logits_attn @ vf
        v_attn = v_attn / (v_attn.norm(dim=-1, keepdim=True) + 1e-8)
        v_attn = v_attn.expand(-1, text_feat.shape[0], -1)
        tf = text_feat.unsqueeze(0).expand(v_attn.shape[0], -1, -1)
        tf = tf + v_attn + self.mlp1(tf + v_attn)

        vf_n = vf / (vf.norm(dim=-1, keepdim=True) + 1e-8)
        tf_n = (tf / (tf.norm(dim=-1, keepdim=True) + 1e-8)).permute(0, 2, 1)
        logits2 = vf_n @ tf_n.type(vf_n.dtype) / 0.07

        scores = torch.sigmoid(logits1.squeeze(-1))
        return scores, logits1, logits2, text_feat

# ══════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════
def apply_uars(w, tau_h=cfg.TAU_H, tau_l=cfg.TAU_L):
    new_w = w.clone()
    new_w[w >= tau_h] = 1.0
    mask = (w > tau_l) & (w < tau_h)
    new_w[mask] = (w[mask] - tau_l) / (tau_h - tau_l + 1e-8)
    new_w[w <= tau_l] = 0.0
    return new_w

def clas2_loss(logits1, labels_bin, lengths, device):
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

def soft_tram_loss(scores, p_teacher, lengths):
    B = scores.shape[0]
    loss = 0.0
    for i in range(B):
        L = max(1, min(int(lengths[i]), scores.shape[1]))
        loss = loss + F.mse_loss(scores[i, :L], p_teacher[i, :L])
    return loss / B

def rmtd_loss(scores, r, y, w):
    B, T = scores.shape
    K2 = r.shape[-1]
    loss = 0.0
    count = 0.0
    for b in range(B):
        s = scores[b]
        max_r = r[b].abs().max() + 1e-8
        for i in range(0, T, 2):
            for k in range(K2):
                if w[b, i, k] <= 0:
                    continue
                offset = -(cfg.K - k) if k < cfg.K else (k - cfg.K) + 1
                j = i + offset
                if j < 0 or j >= T:
                    continue
                r_ij = r[b, i, k]
                y_ij = y[b, i, k].float()
                w_ij = w[b, i, k]
                m_ij = cfg.M_MIN + (cfg.M_MAX - cfg.M_MIN) * (r_ij.abs() / max_r)
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
        w       = apply_uars(torch.cat([nw, aw], 0))

        scores, logits1, logits2, _ = model(visual, lengths, prompt_text)
        labels_cls = get_label_vectors(cats.cpu()).to(device)

        l_c2   = clas2_loss(logits1, labels, lengths, device)
        l_cm   = clasm_loss(logits2, labels_cls, lengths, device)
        l_soft = soft_tram_loss(scores, p, lengths)
        l_rmtd = rmtd_loss(scores, r, y, w) if use_rmtd else torch.tensor(0.0, device=device)

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

        # ===== step-wise evaluation =====
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

    # optional: resume from previous best
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
        # full epoch eval
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