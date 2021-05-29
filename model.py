import keras.losses
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import Progbar
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import config
from dataloader import train_data, val_data
from utils import dice_coef, FocalLoss


class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(ConvBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv1 = layers.Conv2D(self.filters, self.kernel_size, self.strides, padding="same")
        self.conv2 = layers.Conv2D(self.filters, self.kernel_size, self.strides, padding="same")
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, training=None, **kwargs):

        x = self.conv1(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

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
        self.maxpool = layers.MaxPooling2D()
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
        self.concat = layers.Concatenate()
        self.conv_tr_1 = layers.Conv2DTranspose(self.filter_list[3], 2, 2)
        self.conv_tr_2 = layers.Conv2DTranspose(self.filter_list[2], 2, 2)
        self.conv_tr_3 = layers.Conv2DTranspose(self.filter_list[1], 2, 2)
        self.conv_tr_4 = layers.Conv2DTranspose(self.filter_list[0], 2, 2)
        self.out = layers.Conv2D(self.out_channels, kernel_size=1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):

        # Down-sampling Block

        x1 = self.conv_block_1(inputs, training=training)
        downsample_1 = self.maxpool(x1)
        x2 = self.conv_block_2(downsample_1, training=training)
        downsample_2 = self.maxpool(x2)
        x3 = self.conv_block_3(downsample_2, training=training)
        downsample_3 = self.maxpool(x3)
        x4 = self.conv_block_4(downsample_3, training=training)
        downsample_4 = self.maxpool(x4)
        bottle = self.bottleneck(downsample_4, training=training)

        # end of down-sampling

        x = self.pass_thru(bottle)

        # Start of the up-sampling block

        x = self.conv_tr_1(x)
        x = self.concat([x, x4])
        x = self.up_conv_block_1(x, training=training)

        x = self.conv_tr_2(x)
        x = self.concat([x, x3])
        x = self.up_conv_block_2(x, training=training)

        x = self.conv_tr_3(x)
        x = self.concat([x, x2])
        x = self.up_conv_block_3(x, training=training)

        x = self.conv_tr_4(x)
        x = self.concat([x, x1])
        x = self.up_conv_block_4(x, training=training)

        out = self.out(x)

        return out

    # small hack to view the shapes in model.summary()
    # NOTE: shapes arent visible if you create your model via subclassing (use this hack to display them)
    def model(self):
        x = Input(shape=(256, 256, 1))

        return Model(inputs=[x], outputs=self.call(x))

    def get_config(self):
        config = {"name": self.name, "out_channels": self.out_channels, "filters": self.filter_list}
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


if __name__ == "__main__":

    model = Unet(name="unet", out_channels=config.OUT_CHANNELS, filters=config.FILTERS)
    custom_objects = {"Unet": Unet, "ConvBlock": ConvBlock}

    def get_loss(loss_type=config.LOSS_TYPE):
        if loss_type == "focal_loss":
            loss = FocalLoss()
            return loss
        if loss_type == "binary_crossentropy":
            loss = keras.losses.BinaryCrossentropy()
            return loss
        else:
            raise AttributeError

    loss_obj = get_loss(loss_type=config.LOSS_TYPE)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_obj, metrics=[dice_coef])
    print(model.model().summary())

    # Training the model

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/2', histogram_freq=1)
    ckpt = tf.keras.callbacks.ModelCheckpoint(config.CKPT_DIR)
    hist = model.fit(train_data,
                     epochs=10,
                     validation_data=val_data,
                     callbacks=[tensorboard, ckpt])

    model.save("my_model")





