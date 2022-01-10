from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from keras.models import Model

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import optimizers

import numpy as np

# model = VGG16()
# Vgg16Features = Model(inputs=model.inputs, outputs=model.layers[-2].output)


def create_model():
    vgg19 = VGG19(weights='imagenet')
    pretrained = Model(inputs=vgg19.input, outputs=vgg19.get_layer('fc2').output)

    inputs = Input(shape=(224, 224, 4), name='input_17')
    # block 1
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)

    # block 2
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)

    # block 3
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(x)

    # block 4
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(x)

    # block 5
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')(x)

    # block 6
    x = Flatten()(x)
    x = Dense(4096, activation='relu', kernel_initializer='zeros', name='fc1')(x)
    x = Dense(4096, activation='relu', kernel_initializer='zeros', name='fc2')(x)

    vgg_template = Model(inputs=inputs, outputs=x)

    layers_to_modify = ['block1_conv1']  # Turns out the only layer that changes
    # shape due to 4th channel is the first
    # convolution layer.

    for layer in pretrained.layers:  # pretrained Model and template have the same
        # layers, so it doesn't matter which to
        # iterate over.

        if layer.get_weights():  # Skip input, pooling and no weights layers

            target_layer = vgg_template.get_layer(name=layer.name)

            if layer.name in layers_to_modify:

                kernels = layer.get_weights()[0]
                biases = layer.get_weights()[1]

                kernels_extra_channel = np.concatenate((kernels,
                                                        kernels[:,:,-1:,:]),
                                                       axis=-2)  # For channels_last

                target_layer.set_weights([kernels_extra_channel, biases])

            else:
                target_layer.set_weights(layer.get_weights())

    return vgg_template
