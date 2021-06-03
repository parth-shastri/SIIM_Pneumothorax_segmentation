
TRAIN_IMG_DIR = r"C:\Users\shast\datasets\SIIM-train-test\siim\dicom-images-train"
TRAIN_CSV = r"C:\Users\shast\datasets\SIIM-train-test\siim\train-rle.csv"
TEST_IMG_DIR = r"C:\Users\shast\datasets\SIIM-train-test\siim\dicom-images-test"
CKPT_DIR = "model-checkpoint/train-combo-bce-dice(1)"
LOG_DIR = "logs/5-combo-bce-dice(1)"
MODEL_DIR = "my_model_bce_dice"

IMG_SIZE = 128
INIT_IMG_SIZE = 1024
BATCH_SIZE = 8
FILTERS = [64, 128, 256, 512]   # same as in the original paper
OUT_CHANNELS = 1
TRAIN_PERCENT = 0.9

# can be "categorical_crossentropy" or 'focal_loss' or "combo"-weighted{"bce": 3, "dice":1, "focal": 4}
LOSS_TYPE = "combo"
EPOCHS = 30
