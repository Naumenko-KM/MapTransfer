import torch


DATASET_PATH = "C:/Users/supercomp/MapTransfer/dataset_vis2inf/dataset_pix2pix"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = DATASET_PATH + '/train'
VAL_DIR = DATASET_PATH + '/valid'
TEST_DIR = DATASET_PATH + '/test'
EVAL_DIR = 'evaluation'
D_LEARNING_RATE = 1e-3
G_LEARNING_RATE = 1e-3
D_SCHEDULER_GAMMA = 0.5
G_SCHEDULER_GAMMA = 0.5
D_SCHEDULER_STEP = 100
G_SCHEDULER_STEP = 80
BATCH_SIZE = 64
NUM_WORKERS = 0
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 150
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth"
CHECKPOINT_GEN = "gen.pth"
