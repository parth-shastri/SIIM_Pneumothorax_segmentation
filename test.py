import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_io as tfio
import numpy as np
import glob
import os
from model import Unet, ConvBlock, train_data, val_data
from utils import FocalLoss, dice_coef, combo_loss, dice_loss, my_iou_metric
import config
import matplotlib.pyplot as plt
import albumentations as a
from tensorflow.keras import backend as K

model = load_model(config.MODEL_DIR,
                   custom_objects={"Unet": Unet, "ConvBlock": ConvBlock, "FocalLoss": FocalLoss, "dice_coef": dice_coef,
                                   "combo_loss": combo_loss,
                                   "dice_loss": dice_loss,
                                   "iou_metric": my_iou_metric})

test_paths = glob.glob(os.path.join(config.TEST_IMG_DIR, "*/*/*.dcm"))
train_paths = sorted(glob.glob(os.path.join(config.TRAIN_IMG_DIR, "*/*/*.dcm")))

image_bytes = tf.io.read_file(test_paths[0])
image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
image = tf.keras.preprocessing.image.img_to_array(tf.squeeze(image))
image = tf.cast(image, tf.float32)
transform = a.Compose([
    a.Normalize(mean=0, std=1),
    a.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE)
])
transformed = transform(image=image.numpy())
image = transformed["image"]


with tf.device('/CPU:0'):
    for image, mask in val_data.take(4):
        pred_mask = model.predict(image, verbose=1)
        pred_mask = K.sigmoid(pred_mask)

        # Visualize
        print(pred_mask)
        for i in range(len(image)):
            plt.imshow(image[i].numpy(), cmap="bone")
            plt.imshow(mask[i].numpy(), alpha=0.3, cmap="Reds")
            plt.imshow(np.array(np.round(pred_mask[i] > 0.5), dtype=np.float32), alpha=0.3, cmap="Greens")
            plt.show()



