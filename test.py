import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_io as tfio
import numpy as np
import glob
import os
from model import Unet, ConvBlock
from utils import FocalLoss, dice_coef
import config
import matplotlib.pyplot as plt
import albumentations as a

model = load_model("my_model",
                   custom_objects={"Unet": Unet, "ConvBlock": ConvBlock, "FocalLoss": FocalLoss, "dice_coef": dice_coef})

test_paths = glob.glob(os.path.join(config.TEST_IMG_DIR, "*/*/*.dcm"))
train_paths = sorted(glob.glob(os.path.join(config.TRAIN_IMG_DIR, "*/*/*.dcm")))

image_bytes = tf.io.read_file(test_paths[6])
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
    model.load_weights(config.CKPT_DIR)
    pred_mask = model.predict(np.expand_dims(image, axis=0))

# Visualize
print(pred_mask.shape)
plt.imshow(np.array(pred_mask[0] * 255), cmap="gray")
plt.show()
