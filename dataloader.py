import tensorflow as tf
import pydicom
import tensorflow_io as tf_io   # to read dicom files from tensor strings
import matplotlib.pyplot as plt
import pandas as pd
import config
import os
import numpy as np
import glob
import albumentations as a
import tqdm
from sklearn.model_selection import StratifiedKFold

img_height, img_width = config.IMG_SIZE, config.IMG_SIZE

train_rle = pd.read_csv(config.TRAIN_CSV)
train_rle["kfold"] = -1
train_rle['mask'] = 0
train_rle.loc[train_rle[" EncodedPixels"] != '-1', 'mask'] = 1

# Stratified K-fold cross validation for imbalanced labels

str_kfold = StratifiedKFold(n_splits=6)

for idx, (train, val) in enumerate(str_kfold.split(train_rle, y=train_rle["mask"].values)):
    train_rle.loc[val, "kfold"] = idx

neg, pos = np.bincount(train_rle["mask"])
pos_df = train_rle.loc[train_rle["mask"] == 1]
print(f"The no.of POSITIVES :{pos}, The no. of NEGATIVES:{neg}")
print(pos_df.reset_index(drop=True))

train_df = train_rle[train_rle["kfold"] != 0]
val_df = train_rle[train_rle["kfold"] == 0]

train_names = sorted(glob.glob(os.path.join(config.TRAIN_IMG_DIR, "*/*/*.dcm")))
test_names = sorted(glob.glob(os.path.join(config.TEST_IMG_DIR,  "*/*/*.dcm")))
print(f"Found {len(train_names)} images in the train directory.")


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    if rle == ' -1' or rle == '-1':
        return mask.reshape(width, height)
    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def get_mask(encode, width, height):
    if encode == [] or encode == ' -1':
        return rle2mask(' -1', width, height)

    mask = rle2mask(encode[0], width, height)
    for e in encode[1:]:
        mask += rle2mask(e, width, height)
    return mask.T


def get_data(df, train_names):
    train_images = []
    train_mask = []
    c = 0
    for name in tqdm.tqdm(train_names):
        if c > 2000:
            break
        try:

            rle = list(df.loc[df["ImageId"] == '.'.join(name.split('\\')[-1].split('.')[:-1]), " EncodedPixels"].values)

            if not rle:
                continue
            else:
                encoded = get_mask(rle, config.INIT_IMG_SIZE, config.INIT_IMG_SIZE)
                encoded = tf.image.resize(np.expand_dims(encoded, -1), size=[config.IMG_SIZE, config.IMG_SIZE])
                train_mask.append(encoded)
                train_images.append(name)
                c += 1

        except:
            print("no file found")
            continue

    return train_images, train_mask


train_img_path, train_mask = get_data(val_df, train_names)

# val_img_path, val_mask = get_data(val_df, train_names)

# np.save("train_data", (train_img_path, train_mask), dtype="object")
# np.save("val_data", (val_img_path, val_mask), dtype="object")


def preprocess(image_path, mask):  # for train images
    image_bytes = tf.io.read_file(image_path)
    image = tf_io.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
    image = tf.keras.preprocessing.image.img_to_array(tf.squeeze(image))
    # use only for training (AUGMENTATIONS)
    transforms = a.Compose([
        a.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
        a.Normalize(mean=0, std=1),
        a.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8),
        a.OneOf([
            a.RandomGamma(gamma_limit=(90, 110)),
            a.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
        ], p=0.5),
        a.OneOf([
            a.MotionBlur(p=0.2),
            a.MedianBlur(blur_limit=3, p=0.1),
            a.Blur(blur_limit=3, p=0.1),
        ], p=0.3),
    ])

    augmented = transforms(image=image, mask=mask.numpy())
    image = augmented["image"]
    mask = augmented["mask"]
    return tf.cast(image, tf.float32), tf.cast(mask / 255.0, tf.uint16)


def val_pre(image_path, mask):  # for validation images

    image_bytes = tf.io.read_file(image_path)
    image = tf_io.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
    image = tf.keras.preprocessing.image.img_to_array(tf.squeeze(image))
    transforms = a.Compose([
        a.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
        a.Normalize(mean=0, std=1)
    ])
    augmented = transforms(image=image, mask=mask.numpy())
    image = augmented["image"]
    mask = augmented["mask"]
    return tf.cast(image, tf.float32), tf.cast(mask / 255.0, tf.uint16)


# map this to the tf.py_func mapped dataset to restored shape
def set_shapes(img, label, img_shape=(config.IMG_SIZE, config.IMG_SIZE, 1)):
    img.set_shape(img_shape)
    label.set_shape(img_shape)
    return img, label


train_tensor = tf.data.Dataset.from_tensor_slices((train_img_path, train_mask)).shuffle(1800)

# val_tensor = tf.data.Dataset.from_tensor_slices((val_img_path, val_mask))

tf.random.set_seed(42)

train_data = train_tensor.map(
    lambda x, y: tf.py_function(preprocess, [x, y], [tf.float32, tf.uint16]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).map(
    set_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE
).take(int(0.9 * 2000)).batch(config.BATCH_SIZE)

val_data = train_tensor.map(
    lambda x, y: tf.py_function(val_pre, [x, y], [tf.float32, tf.uint16]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).map(set_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE
      ).skip(int(0.9 * 2000)).batch(config.BATCH_SIZE)

if __name__ == "__main__":

    '''Test Code'''
    # img_ex = pydicom.read_file(train_img_path[6]).pixel_array
    # img_ex = tf.image.resize(img_ex.reshape((config.INIT_IMG_SIZE, config.INIT_IMG_SIZE, -1)),
    #                          size=(config.IMG_SIZE, config.IMG_SIZE))
    # plt.imshow(img_ex, cmap="gray")
    # plt.imshow(train_mask[6].numpy().reshape(config.IMG_SIZE, -1), alpha=0.25)
    # plt.show()
    for n, (images, masks) in enumerate(val_data.take(5)):
        for i, mask in enumerate(masks):
            if mask.numpy().any() == 1.:
                plt.imshow(images[i], cmap="gray")
                plt.imshow(mask, alpha=0.3)
                plt.show()
                # print(mask, images.shape)
                print("batch:", n, '\t', "the index : ", i)

            else:
                continue









