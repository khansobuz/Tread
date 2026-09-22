from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from scipy.spatial.distance import pdist, squareform
from config import cfg

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
# MODEL  (Lightweight CLIP Student Model – LCSM)
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
        scores = torch.sigmoid(logits1.squeeze(-1))   # S_θ(c_i) ∈ [0,1]
        return scores, logits1, logits2, text_feat