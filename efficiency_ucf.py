"""
Efficiency evaluation for UCF:
- Params (M)
- GPU Memory (GB)
- Time (s/video)

Models:
1) Ours Lightweight Student
2) MLLM Qwen2.5-VL-7B
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import warnings
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image

warnings.filterwarnings("ignore")

# ====================== PATHS (EDIT IF NEEDED) ======================
MLLM_PATH = "/home/sabuj_khan/research/projects/DMN/Qwen_7B"   # or Qwen2.5-VL-7B-Instruct
# One sample UCF video for timing
SAMPLE_VIDEO = "/home/sabuj_khan/research/projects/DMN/UCF_Crime/Videos/Abuse/Abuse001_x264.mp4"

# Optional: path to your trained student checkpoint
STUDENT_CKPT = "checkpoint/best_vadclip_tram.pth"   # set None if not available

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 2
REPEAT = 5
CLIP_LEN = 16
T = 256
FEAT_DIM = 512

# ====================== STUDENT MODEL (lightweight, same style as yours) ======================
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

    def attention(self, x):
        am = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=am)[0]

    def forward(self, x):
        x, pm = x
        x = x + self.attention(self.ln_1(x))
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

class StudentModel(nn.Module):
    """Lightweight student used at inference (no MLLM)."""
    def __init__(self, T=256, width=512, layers=2, heads=1, attn_window=8):
        super().__init__()
        mask = torch.empty(T, T).fill_(float("-inf"))
        for i in range(int(T / attn_window)):
            s = i * attn_window
            e = min(s + attn_window, T)
            mask[s:e, s:e] = 0
        self.temporal = Transformer(width, layers, heads, attn_mask=mask)
        self.pos_embed = nn.Embedding(T, width)
        nn.init.normal_(self.pos_embed.weight, std=0.01)
        self.classifier = nn.Sequential(
            nn.Linear(width, width),
            QuickGELU(),
            nn.Linear(width, 1)
        )
        self.T = T

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        pos = self.pos_embed(torch.arange(T, device=x.device)).unsqueeze(0)
        x = x + pos
        x = x.permute(1, 0, 2)
        x, _ = self.temporal((x, None))
        x = x.permute(1, 0, 2)
        scores = torch.sigmoid(self.classifier(x).squeeze(-1))
        return scores


def count_params_m(model):
    n = sum(p.numel() for p in model.parameters())
    return n / 1e6


def gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


# ====================== STUDENT EFFICIENCY ======================
def eval_student():
    print("\n" + "=" * 60)
    print("OURS: Lightweight Student")
    print("=" * 60)

    reset_peak_mem()
    model = StudentModel().to(DEVICE).eval()

    # optional load checkpoint (ignore mismatch)
    if STUDENT_CKPT and os.path.exists(STUDENT_CKPT):
        try:
            ckpt = torch.load(STUDENT_CKPT, map_location=DEVICE)
            state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint: {STUDENT_CKPT}")
        except Exception as e:
            print(f"Checkpoint not fully loaded (ok for efficiency): {e}")

    params_m = count_params_m(model)
    print(f"Params (M): {params_m:.2f}")

    # dummy feature input like UCF CLIP features
    x = torch.randn(1, T, FEAT_DIM, device=DEVICE)

    # warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    reset_peak_mem()
    times = []
    with torch.no_grad():
        for _ in range(REPEAT):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - t0)

    mem = gpu_mem_gb()
    t_mean = float(np.mean(times))
    print(f"GPU Mem (GB): {mem:.2f}")
    print(f"Time (s/video): {t_mean:.4f}   [feature-level inference]")
    return {
        "name": "Ours Lightweight Student",
        "params_m": params_m,
        "gpu_mem_gb": mem,
        "time_s_video": t_mean
    }


# ====================== MLLM EFFICIENCY ======================
def read_video_clips(video_path, clip_len=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    clips = []
    for i in range(0, max(0, len(frames) - clip_len + 1), clip_len):
        clips.append(frames[i:i + clip_len])
    return clips


def eval_mllm():
    print("\n" + "=" * 60)
    print("MLLM: Qwen2.5-VL-7B")
    print("=" * 60)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    reset_peak_mem()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MLLM_PATH,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MLLM_PATH, trust_remote_code=True)
    model.eval()

    params_m = count_params_m(model)
    print(f"Params (M): {params_m:.2f}")

    if not os.path.exists(SAMPLE_VIDEO):
        print(f"[WARN] SAMPLE_VIDEO not found: {SAMPLE_VIDEO}")
        print("Using synthetic image timing only.")
        clips = [np.random.randint(0, 255, (CLIP_LEN, 224, 224, 3), dtype=np.uint8)]
    else:
        clips = read_video_clips(SAMPLE_VIDEO, CLIP_LEN)
        if len(clips) == 0:
            clips = [np.random.randint(0, 255, (CLIP_LEN, 224, 224, 3), dtype=np.uint8)]
        print(f"Sample video clips: {len(clips)}")

    def run_one_clip(clip_frames):
        mid = len(clip_frames) // 2
        image = Image.fromarray(clip_frames[mid])
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text":
                 "Rate how anomalous this scene is from 0 to 1. "
                 "Reply with only one number between 0 and 1."}
            ]
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=8)

    # warmup on first clip
    for _ in range(WARMUP):
        run_one_clip(clips[0])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # measure full-video time (all clips of sample video)
    reset_peak_mem()
    times = []
    n_clips = min(len(clips), 30)   # cap for faster benchmark; set None for full
    use_clips = clips[:n_clips]

    for _ in range(REPEAT):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for c in use_clips:
            run_one_clip(c)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    mem = gpu_mem_gb()
    t_mean = float(np.mean(times))
    # scale to full video if capped
    if n_clips < len(clips):
        t_mean = t_mean * (len(clips) / n_clips)

    print(f"GPU Mem (GB): {mem:.2f}")
    print(f"Time (s/video): {t_mean:.2f}   [raw-video MLLM teacher path]")
    print(f"(measured over {len(clips)} clips; internal probe used {n_clips} clips)")
    return {
        "name": "MLLM Qwen2.5-VL-7B",
        "params_m": params_m,
        "gpu_mem_gb": mem,
        "time_s_video": t_mean
    }


# ====================== MAIN ======================
def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []
    results.append(eval_student())

    # free student before loading huge MLLM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results.append(eval_mllm())

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Method':<28} {'Params (M)':>10} {'GPU Mem (GB)':>12} {'Time (s/video)':>14}")
    print("-" * 68)
    for r in results:
        print(f"{r['name']:<28} {r['params_m']:>10.2f} {r['gpu_mem_gb']:>12.2f} {r['time_s_video']:>14.4f}")
    print("=" * 60)
    print("Note:")
    print("- Student time = inference on CLIP features (deployment setting)")
    print("- MLLM time = teacher path on raw video clips (offline supervision)")


if __name__ == "__main__":
    main()