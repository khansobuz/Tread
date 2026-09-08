import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

# ====================== PATHS (XD-Violence) ======================
MLLM_PATH = "/home/sabuj_khan/research/projects/DMN/Qwen_7B"

ROOT_VIDEOS = "/home/sabuj_khan/research/projects/DMN/XD_V/Videos"

TRAIN_TXT = "/home/sabuj_khan/research/projects/DMN/XD_V/splits/train_list.txt"
TEST_TXT  = "/home/sabuj_khan/research/projects/DMN/XD_V/splits/test_list.txt"

SAVE_DIR_TRAIN = "/home/sabuj_khan/research/projects/DMN/tram_outputs_xd/train"
SAVE_DIR_TEST  = "/home/sabuj_khan/research/projects/DMN/tram_outputs_xd/test"

os.makedirs(SAVE_DIR_TRAIN, exist_ok=True)
os.makedirs(SAVE_DIR_TEST, exist_ok=True)

# ====================== SETTINGS ======================
CLIP_LEN = 16
K = 2
DELTA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ====================== LOAD MODEL ======================
print("\nLoading Qwen2.5-VL...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MLLM_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MLLM_PATH, trust_remote_code=True)

model.eval()
for param in model.parameters():
    param.requires_grad = False

print("MLLM loaded and frozen successfully!\n")

# ====================== HELPERS ======================
def read_video_clips(video_path, clip_len=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    clips = []
    for i in range(0, len(frames) - clip_len + 1, clip_len):
        clips.append(frames[i:i + clip_len])
    return clips


def ask_mllm_anomaly_score(clip_frames):
    mid = len(clip_frames) // 2
    image = Image.fromarray(clip_frames[mid])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text":
                 "This is a frame from a video that may contain violence or abnormal events. "
                 "Rate how anomalous / violent this scene is from 0 to 1. "
                 "0 = completely normal, 1 = highly anomalous (fighting, shooting, explosion, riot, accident...). "
                 "Reply with only one number between 0 and 1."}
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    try:
        score = float(output_text.strip())
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5
    return score


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or frame_count <= 0:
        return None, None, None

    duration_sec = frame_count / fps
    file_size_gb = os.path.getsize(video_path) / (1024 ** 3)
    return duration_sec, file_size_gb, fps


def find_video_path(video_name, split="train"):
    video_name = os.path.basename(video_name.strip())

    if split.lower() == "train":
        candidates = [
            os.path.join(ROOT_VIDEOS, "Train", video_name),
            os.path.join(ROOT_VIDEOS, "train", video_name),
        ]
    else:
        candidates = [
            os.path.join(ROOT_VIDEOS, "Test", video_name),
            os.path.join(ROOT_VIDEOS, "test", video_name),
        ]

    for path in candidates:
        if os.path.exists(path):
            return path

    search_root = os.path.join(ROOT_VIDEOS, "Train" if split.lower() == "train" else "Test")
    if not os.path.exists(search_root):
        search_root = ROOT_VIDEOS

    for root, dirs, files in os.walk(search_root):
        if video_name in files:
            return os.path.join(root, video_name)

    return None


def process_one_video(video_name, save_dir, split="train"):
    video_name = video_name.strip()
    base = os.path.basename(video_name)

    if base.lower().endswith((".mp4", ".avi", ".mkv")):
        save_name = os.path.splitext(base)[0] + ".pt"
    else:
        save_name = base + ".pt"
    save_path = os.path.join(save_dir, save_name)

    if os.path.exists(save_path):
        print(f"⏭️ Already exists: {save_name}")
        return True

    video_path = find_video_path(video_name, split=split)
    if video_path is None:
        print(f"❌ Not found: {video_name}")
        return False

    duration_sec, file_size_gb, fps = get_video_info(video_path)
    if duration_sec is None:
        print(f"❌ Cannot read video info: {video_name}")
        return False

    duration_min = duration_sec / 60.0

    # skip too large / too long
    if file_size_gb > 1.0:
        print(f"⏭️ SKIPPED (size {file_size_gb:.2f} GB)")
        return True

    if duration_min > 60:
        print(f"⏭️ SKIPPED (duration {duration_min:.1f} min)")
        return True

    clips = read_video_clips(video_path, CLIP_LEN)
    N = len(clips)
    if N == 0:
        print(f"❌ No clips")
        return False

    p = []
    for clip in clips:
        score = ask_mllm_anomaly_score(clip)
        p.append(score)

    p = torch.tensor(p, dtype=torch.float32)

    # temporal relations (TRAM)
    r_list, y_list = [], []
    for i in range(N):
        rel_r, rel_y = [], []
        for offset in range(-K, K + 1):
            if offset == 0:
                continue
            j = i + offset
            if j < 0 or j >= N:
                rel_r.append(0.0)
                rel_y.append(0)
                continue

            r_ij = (p[i] - p[j]).item()
            y_ij = 1 if r_ij > DELTA else (-1 if r_ij < -DELTA else 0)
            rel_r.append(r_ij)
            rel_y.append(y_ij)

        r_list.append(rel_r)
        y_list.append(rel_y)

    r = torch.tensor(r_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    w = torch.ones_like(r)

    torch.save(
        {
            "video_name": video_name,
            "p": p,
            "r": r,
            "y": y,
            "w": w,
        },
        save_path
    )

    print(f"✅ {save_name} | clips: {N} | p: {p.shape}")
    return True


# ====================== RUN ======================
print("Reading lists...")
with open(TRAIN_TXT) as f:
    train_list = [line.strip() for line in f if line.strip()]

with open(TEST_TXT) as f:
    test_list = [line.strip() for line in f if line.strip()]

print(f"Train videos: {len(train_list)}")
print(f"Test videos : {len(test_list)}")

# ---------- TRAIN ----------
print("\n" + "=" * 50)
print("TRAIN SET (XD-Violence)")
print("=" * 50)

success = 0
for idx, video_name in enumerate(train_list, 1):
    print(f"[{idx}/{len(train_list)}] {os.path.basename(video_name)} ... ", end="", flush=True)
    try:
        if process_one_video(video_name, SAVE_DIR_TRAIN, split="train"):
            success += 1
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\nTrain done: {success}/{len(train_list)} successful")

# ---------- TEST ----------
print("\n" + "=" * 50)
print("TEST SET (XD-Violence)")
print("=" * 50)

success = 0
for idx, video_name in enumerate(test_list, 1):
    print(f"[{idx}/{len(test_list)}] {os.path.basename(video_name)} ... ", end="", flush=True)
    try:
        if process_one_video(video_name, SAVE_DIR_TEST, split="test"):
            success += 1
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\nTest done: {success}/{len(test_list)} successful")
print("\n===== ALL DONE (XD-Violence TRAM) =====")