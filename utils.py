import tensorflow as tf


def dice_coef(y_true, y_pred):

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_true, axis=[1, 2, 3])

    metric = (2*intersection) / (union + 1e-6)
    return tf.reduce_mean(metric)
