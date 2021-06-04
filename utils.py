import tensorflow as tf
from tensorflow import keras
import config
from tensorflow.keras import backend as K
import numpy as np


def get_iou_vector(A, B):
    # Numpy version
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue

        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union

        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10

        metric += iou

    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_function(get_iou_vector, [label, (K.cast(K.sigmoid(pred) > 0.1, tf.float32))], tf.float64)


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.cast(K.flatten(y_true), tf.float32)
    y_pred_f = K.flatten(K.sigmoid(y_pred))
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def dice_coef(y_true, y_pred):

    y_true_f = tf.cast(K.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(K.sigmoid(K.flatten(y_pred)), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_true_f) + K.sum(y_pred_f)

    if K.sum(y_true_f) == 0 and K.sum(y_pred_f) == 0:
        return 1.0

    score = (2*intersection) / (union + 1e-6)
    return tf.reduce_mean(score)


class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.75, gamma=2., from_logits=False, name="focal_loss", **kwargs):
        super(FocalLoss, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def __call__(self, y_true, y_pred):
        bce = keras.losses.BinaryCrossentropy(from_logits=self.from_logits,
                                              reduction=tf.keras.losses.Reduction.NONE)(y_true,
                                                                                        y_pred,
                                                                                        sample_weight=[self.alpha])
        # a_t = tf.where(y_true == 1, self.alpha, 1 - self.alpha)
        p_t = tf.exp(-tf.expand_dims(bce, axis=-1))
        focal_loss = ((1 - p_t) ** self.gamma) * tf.expand_dims(bce, axis=-1)
        return focal_loss

    def get_config(self):
        config = super(FocalLoss, self).from_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def combo_loss(y_true, y_pred):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred) + dice_loss(y_true, y_pred)
           # 4 * FocalLoss(from_logits=True, alpha=4.)(y_true, y_pred)
    return loss


if __name__ == "__main__":

    """Test Code"""

    y_true = tf.ones((4, config.IMG_SIZE, config.IMG_SIZE, 1))
    y_pred = tf.random.normal((4, config.IMG_SIZE, config.IMG_SIZE, 1))

    loss_obj = FocalLoss(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_obj(y_true, y_pred)
    bce_loss = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    print(loss)

