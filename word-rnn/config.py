# config.py

# ========= data =========
DATA_PATH = "data.txt"

# ========= model =========
HIDDEN_SIZE = 256
NUM_LAYERS = 1

# ========= training =========
BATCH_SIZE = 1        # we are not batching yet
SEQ_LEN = 20
LR = 0.003
EPOCHS = 20

# ========= sampling =========
TEMPERATURE = 0.8
MAX_GEN_LEN = 30

# ========= device =========
DEVICE = "cuda"      # auto-fallback to cpu
