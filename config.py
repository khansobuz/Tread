import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    K              = 2                 # temporal context radius (paper Eq. neighborhood)
    M_MIN          = 0.05              # RMTD min margin
    M_MAX          = 0.35              # RMTD max margin
    LAMBDA_SOFT    = 0.40              # weight for soft TRAM (MSE on p)
    LAMBDA_R       = 0.05              # weight for RMTD loss
    RMTD_END_EP    = 12                # turn off RMTD after this epoch
    TAU_H          = 0.55              # UARS high reliability threshold
    TAU_L          = 0.20              # UARS low reliability threshold
    EPOCHS         = 25
    BATCH_SIZE     = 16
    LR             = 5e-5
    WEIGHT_DECAY   = 5e-4
    EVAL_EVERY     = 30                # step-wise eval
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