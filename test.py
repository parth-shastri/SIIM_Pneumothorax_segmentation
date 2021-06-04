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
                                   "my_iou_metric": my_iou_metric}
                   )

# test_paths = glob.glob(os.path.join(config.TEST_IMG_DIR, "*/*/*.dcm"))
# train_paths = sorted(glob.glob(os.path.join(config.TRAIN_IMG_DIR, "*/*/*.dcm")))
#
# image_bytes = tf.io.read_file(test_paths[0])
# image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
# image = tf.keras.preprocessing.image.img_to_array(tf.squeeze(image))
# image = tf.cast(image, tf.float32)
# transform = a.Compose([
#     a.Normalize(mean=0, std=1),
#     a.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE)
# ])
# transformed = transform(image=image.numpy())
# image = transformed["image"]
#

with tf.device('/CPU:0'):

    thresh = 0.5
    batch = 64
    grid_width = 16
    grid_height = 4
    fig, axes = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))

    inf_data = val_data.take(8)
    model.load_weights(config.CKPT_DIR)
    pred_logits = model.predict(inf_data, verbose=1).reshape((-1, config.IMG_SIZE, config.IMG_SIZE, 1))
    pred_masks = tf.sigmoid(pred_logits)

    images, masks = [], []
    for (imgs, msks) in inf_data:
        images.append(imgs)
        masks.append(msks)

    images = np.array(images).reshape((-1, config.IMG_SIZE, config.IMG_SIZE, 1))
    masks = np.array(masks).reshape((-1, config.IMG_SIZE, config.IMG_SIZE, 1))

    for i, (im, mask) in enumerate(zip(images, masks)):
        pred = pred_masks[i]
        ax = axes[int(i / grid_width), i % grid_width]
        ax.imshow(im[..., 0], cmap="bone")
        ax.imshow(mask, cmap="Reds", alpha=0.5)
        ax.imshow(np.array(np.round(pred > thresh), dtype=np.float32), cmap="Greens", alpha=0.5)
        ax.axis("off")
    plt.show()

