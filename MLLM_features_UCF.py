import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

# ====================== PATHS ======================
MLLM_PATH   = "/home/sabuj_khan/research/projects/DMN/Qwen_7B"
ROOT_VIDEOS = "/home/sabuj_khan/research/projects/DMN/UCF_Crime/Videos"
TRAIN_TXT   = "/home/sabuj_khan/research/projects/DMN/UCF_Crime/Anomaly_Detection_splits/Anomaly_Train.txt"
TEST_TXT    = "/home/sabuj_khan/research/projects/DMN/UCF_Crime/Anomaly_Detection_splits/Anomaly_Test.txt"

SAVE_DIR_TRAIN = "/home/sabuj_khan/research/projects/DMN/tram_outputs/train"
SAVE_DIR_TEST  = "/home/sabuj_khan/research/projects/DMN/tram_outputs/test"

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
print("\nLoading Qwen2.5-VL-7B-Instruct...")

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
        clips.append(frames[i:i+clip_len])
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
                 "This is a frame from a surveillance video. "
                 "Rate how anomalous this scene is from 0 to 1. "
                 "0 = completely normal, 1 = highly anomalous (crime, violence, accident...). "
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


def process_one_video(video_name, save_dir):
    video_name = video_name.strip()

    # ---------- SKIP ALREADY PROCESSED ----------
    save_name = os.path.basename(video_name).replace(".mp4", ".pt")
    save_path = os.path.join(save_dir, save_name)

    if os.path.exists(save_path):
        print(f"⏭️ Already exists: {save_name}")
        return True

    # ---------- FIND VIDEO ----------
    possible_paths = [
        os.path.join(ROOT_VIDEOS, video_name),
        os.path.join(ROOT_VIDEOS, os.path.basename(video_name)),
        os.path.join(ROOT_VIDEOS, video_name.replace("/", os.sep)),
    ]

    video_path = None

    for path in possible_paths:
        if os.path.exists(path):
            video_path = path
            break

    if video_path is None:
        target = os.path.basename(video_name)

        for root, dirs, files in os.walk(ROOT_VIDEOS):
            if target in files:
                video_path = os.path.join(root, target)
                break

    if video_path is None:
        print(f"❌ Not found: {video_name}")
        return False

    # ==========================================================
    # CHECK VIDEO SIZE AND DURATION BEFORE LOADING FRAMES
    # ==========================================================

    duration_sec, file_size_gb, fps = get_video_info(video_path)

    if duration_sec is None:
        print(f"❌ Cannot read video info: {video_name}")
        return False

    duration_min = duration_sec / 60

    print(
        f"Size: {file_size_gb:.2f} GB | "
        f"Duration: {duration_min:.1f} min"
    )

    # ---------- SKIP LARGE VIDEOS ----------
    if file_size_gb > 1.0:
        print(
            f"⏭️ SKIPPED: {save_name} "
            f"(size {file_size_gb:.2f} GB > 1 GB)"
        )
        return True

    # ---------- SKIP LONG VIDEOS ----------
    if duration_min > 60:
        print(
            f"⏭️ SKIPPED: {save_name} "
            f"(duration {duration_min:.1f} min > 20 min)"
        )
        return True

    # ==========================================================
    # PROCESS NORMAL-SIZED VIDEO
    # ==========================================================

    clips = read_video_clips(video_path, CLIP_LEN)
    N = len(clips)

    p = []

    for clip in clips:
        score = ask_mllm_anomaly_score(clip)
        p.append(score)

    p = torch.tensor(p, dtype=torch.float32)

    # ---------- TEMPORAL RELATIONS ----------
    r_list, y_list = [], []

    for i in range(N):

        rel_r = []
        rel_y = []

        for offset in range(-K, K + 1):

            if offset == 0:
                continue

            j = i + offset

            if j < 0 or j >= N:
                rel_r.append(0.0)
                rel_y.append(0)
                continue

            r_ij = (p[i] - p[j]).item()

            y_ij = (
                1 if r_ij > DELTA
                else (-1 if r_ij < -DELTA else 0)
            )

            rel_r.append(r_ij)
            rel_y.append(y_ij)

        r_list.append(rel_r)
        y_list.append(rel_y)

    r = torch.tensor(r_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)

    w = torch.ones_like(r)

    # ---------- SAVE ----------
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

    print(
        f"✅ {save_name} | "
        f"clips: {N} | "
        f"p: {p.shape}"
    )

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
print("\n" + "="*50)
print("TRAIN SET")
print("="*50)

success = 0
for idx, video_name in enumerate(train_list, 1):
    print(f"[{idx}/{len(train_list)}] {os.path.basename(video_name)} ... ", end="", flush=True)
    try:
        if process_one_video(video_name, SAVE_DIR_TRAIN):
            success += 1
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\nTrain done: {success}/{len(train_list)} successful")

# ---------- TEST ----------
print("\n" + "="*50)
print("TEST SET")
print("="*50)

success = 0
for idx, video_name in enumerate(test_list, 1):
    print(f"[{idx}/{len(test_list)}] {os.path.basename(video_name)} ... ", end="", flush=True)
    try:
        if process_one_video(video_name, SAVE_DIR_TEST):
            success += 1
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\nTest done: {success}/{len(test_list)} successful")
print("\n===== ALL DONE =====")
