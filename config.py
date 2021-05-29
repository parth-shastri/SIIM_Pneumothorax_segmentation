
TRAIN_IMG_DIR = r"C:\Users\shast\datasets\SIIM-train-test\siim\dicom-images-train"
TRAIN_CSV = r"C:\Users\shast\datasets\SIIM-train-test\siim\train-rle.csv"
TEST_IMG_DIR = r"C:\Users\shast\datasets\SIIM-train-test\siim\dicom-images-test"
CKPT_DIR = "model-checkpoint/train-focal"

IMG_SIZE = 256
INIT_IMG_SIZE = 1024
BATCH_SIZE = 4
FILTERS = [64, 128, 256, 512]   # same as in the original paper
OUT_CHANNELS = 1
VAL_PERCENT = 0.1

LOSS_TYPE = 'focal_loss'   # can be "binary_crossentropy"
