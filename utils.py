import tensorflow as tf
from tensorflow import keras

import config


def dice_coef(y_true, y_pred):

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_true, axis=[1, 2, 3])

    metric = (2*intersection) / (union + 1e-6)
    return tf.reduce_mean(metric)


class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.75, gamma=2., from_logits=False, name="focal_loss", **kwargs):
        super(FocalLoss, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        bce = keras.losses.SparseCategoricalCrossentropy(from_logits=self.from_logits,
                                                         reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
        a_t = tf.where(y_true == 1, self.alpha, 1 - self.alpha)
        p_t = tf.exp(-tf.expand_dims(bce, axis=-1))
        focal_loss = a_t * ((1 - p_t) ** self.gamma) * tf.expand_dims(bce, axis=-1)
        return focal_loss

    def get_config(self):
        config = super(FocalLoss, self).from_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":

    """Test Code"""

    y_true = tf.ones((4, config.IMG_SIZE, config.IMG_SIZE, 1))
    y_pred = tf.random.normal((4, config.IMG_SIZE, config.IMG_SIZE, 1))

    loss_obj = FocalLoss(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_obj(y_true, y_pred)
    bce_loss = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
    print(loss)

