import keras.losses
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import Progbar
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import config
from dataloader import train_data, val_data
from utils import dice_coef, FocalLoss, combo_loss, my_iou_metric
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'


class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv1 = layers.Conv2D(self.filters, self.kernel_size, self.strides, padding="same")
        self.conv2 = layers.Conv2D(self.filters, self.kernel_size, self.strides, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.relu2 = layers.ReLU()

    def call(self, inputs, training=None, **kwargs):

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = layers.Dropout(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        out = layers.Dropout(0.2)(x)

        return out

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({"filters": self.filters,
                       "kernel_size": self.kernel_size,
                       "strides": self.strides})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Unet(Model):
    def __init__(self, name, out_channels=1, filters=(64, 128, 256, 512)):
        super(Unet, self).__init__(name=name)
        self.filter_list = filters
        self.out_channels = out_channels
        self.maxpool1 = layers.MaxPooling2D(name="pool_1")
        self.maxpool2 = layers.MaxPooling2D(name="pool_2")
        self.maxpool3 = layers.MaxPooling2D(name="pool_3")
        self.maxpool4 = layers.MaxPooling2D(name="pool_4")
        self.conv_block_1 = ConvBlock(self.filter_list[0], 3, 1)
        self.conv_block_2 = ConvBlock(self.filter_list[1], 3, 1)
        self.conv_block_3 = ConvBlock(self.filter_list[2], 3, 1)
        self.conv_block_4 = ConvBlock(self.filter_list[3], 3, 1)
        self.bottleneck = layers.Conv2D(1024, kernel_size=3, strides=1, padding="same")
        self.pass_thru = layers.Conv2D(512, kernel_size=3, strides=1, padding='same')
        self.up_conv_block_1 = ConvBlock(self.filter_list[3], 3, 1)
        self.up_conv_block_2 = ConvBlock(self.filter_list[2], 3, 1)
        self.up_conv_block_3 = ConvBlock(self.filter_list[1], 3, 1)
        self.up_conv_block_4 = ConvBlock(self.filter_list[0], 3, 1)
        self.concat1 = layers.Concatenate(name="concat_1")
        self.concat2 = layers.Concatenate(name="concat_2")
        self.concat3 = layers.Concatenate(name="concat_3")
        self.concat4 = layers.Concatenate(name="concat_4")
        self.conv_tr_1 = layers.Conv2DTranspose(self.filter_list[3], 2, 2, padding="same", activation="relu")
        self.conv_tr_2 = layers.Conv2DTranspose(self.filter_list[2], 2, 2, padding="same", activation="relu")
        self.conv_tr_3 = layers.Conv2DTranspose(self.filter_list[1], 2, 2, padding="same", activation="relu")
        self.conv_tr_4 = layers.Conv2DTranspose(self.filter_list[0], 2, 2, padding="same", activation="relu")
        self.out = layers.Conv2D(self.out_channels, kernel_size=1, strides=1, name="prediction_layer")

    def call(self, inputs, training=None, mask=None):

        # Down-sampling Block

        x1 = self.conv_block_1(inputs, traning=training)  # 128
        downsample_1 = self.maxpool1(x1)
        x2 = self.conv_block_2(downsample_1, training=training)  # 64
        downsample_2 = self.maxpool2(x2)
        x3 = self.conv_block_3(downsample_2, training=training)  # 32
        downsample_3 = self.maxpool3(x3)
        # x4 = self.conv_block_4(downsample_3)  # 16
        # downsample_4 = self.maxpool4(x4)
        bottle = self.bottleneck(downsample_3)  # 8

        # end of down-sampling

        x = self.pass_thru(bottle)

        # Start of the up-sampling block

        x = self.conv_tr_1(x)
        x = self.concat1([x3, x])
        x = self.up_conv_block_1(x,training=training)

        x = self.conv_tr_2(x)
        x = self.concat2([x2, x])
        x = self.up_conv_block_2(x, training=training)

        x = self.conv_tr_3(x)
        x = self.concat3([x1, x])
        x = self.up_conv_block_3(x, training=training)

        # x = self.conv_tr_4(x)
        # x = self.concat4([x, x1])
        # x = self.up_conv_block_4(x)

        x = layers.Dropout(0.5)(x)

        out = self.out(x)

        return out

    # small hack to view the shapes in model.summary()
    # NOTE: shapes arent visible if you create your model via subclassing (use this hack to display them)
    def model(self):
        x = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 1))

        return Model(inputs=[x], outputs=self.call(x))

    def get_config(self):
        config = {"name": self.name, "out_channels": self.out_channels, "filters": self.filter_list}
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


def get_loss(loss_type=config.LOSS_TYPE):
    if loss_type == "focal_loss":
        loss = FocalLoss(from_logits=True)
        return loss
    if loss_type == "categorical_crossentropy":
        loss = keras.losses.BinaryCrossentropy(from_logits=True)
        return loss
    if loss_type == "combo":
        loss = combo_loss
        return loss
    else:
        raise AttributeError


def get_model(img_size, num_classes):
    inputs = Input(shape=img_size+(1,))
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    res1 = x
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    res2 = x
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)

    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    res3 = x
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)

    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(256, 2, padding='same', activation='relu')(x)
    x = layers.concatenate([res3, x], axis=3)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 2, padding='same', activation='relu')(x)
    x = layers.concatenate([res2, x], axis=3)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 2, padding='same', activation='relu')(x)
    x = layers.concatenate([res1, x], axis=3)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    outputs = layers.Conv2D(num_classes, 1, padding='same')(x)

    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":

    model = Unet(name="Unet", out_channels=1, filters=config.FILTERS)
    custom_objects = {"Unet": Unet, "ConvBlock": ConvBlock}

    loss_obj = get_loss(loss_type=config.LOSS_TYPE)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_obj,
                  metrics=[my_iou_metric, dice_coef])
    print(model.model().summary())

    # Training the model :

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.LOG_DIR, histogram_freq=1)
    ckpt = tf.keras.callbacks.ModelCheckpoint(config.CKPT_DIR)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    model.load_weights(config.CKPT_DIR)
    hist = model.fit(train_data,
                     epochs=config.EPOCHS,
                     validation_data=val_data,
                     callbacks=[tensorboard, ckpt, es],
                     verbose=1)

    model.save(config.MODEL_DIR)
    #
    model.save_weights("model_combo_unet_40.h5")




