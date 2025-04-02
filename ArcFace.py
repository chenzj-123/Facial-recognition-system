from tensorflow import keras
from tensorflow.python.keras import backend
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
import tensorflow


import os
from pathlib import Path
import gdown


#url = "https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY"

def loadModel(url='https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5'):
    base_model = ResNet34()
    inputs = base_model.inputs[0]
    x = base_model.outputs[0]  # shape: (None, 25088)

    x = keras.layers.Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(x)
    embedding = keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(x)

    model = keras.models.Model(inputs, embedding, name='ResNet34')

    weights_path = "arcface_weights.h5"
    if not os.path.isfile(weights_path):
        print("Downloading arcface_weights.h5...")
        gdown.download(url, weights_path, quiet=False)

    # ✅ 加载权重时避免层冲突
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
	bn_axis = 3

	if conv_shortcut:
		shortcut = tensorflow.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False, kernel_initializer='glorot_normal', name=name + '_0_conv')(x)
		shortcut = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_0_bn')(shortcut)
	else:
		shortcut = x

	x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_1_bn')(x)
	x = tensorflow.keras.layers.ZeroPadding2D(padding=1, name=name + '_1_pad')(x)
	x = tensorflow.keras.layers.Conv2D(filters, 3, strides=1, kernel_initializer='glorot_normal', use_bias=False, name=name + '_1_conv')(x)
	x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_2_bn')(x)
	x = tensorflow.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_1_prelu')(x)

	x = tensorflow.keras.layers.ZeroPadding2D(padding=1, name=name + '_2_pad')(x)
	x = tensorflow.keras.layers.Conv2D(filters, kernel_size, strides=stride, kernel_initializer='glorot_normal', use_bias=False, name=name + '_2_conv')(x)
	x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_3_bn')(x)

	x = tensorflow.keras.layers.Add(name=name + '_add')([shortcut, x])
	return x

def stack1(x, filters, blocks, stride1=2, name=None):
	x = block1(x, filters, stride=stride1, name=name + '_block1')
	for i in range(2, blocks + 1):
		x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
	return x

def stack_fn(x):
	x = stack1(x, 64, 3, name='conv2')
	x = stack1(x, 128, 4, name='conv3')
	x = stack1(x, 256, 6, name='conv4')
	return stack1(x, 512, 3, name='conv5')

def ResNet34():
    img_input = keras.layers.Input(shape=(112, 112, 3))
    x = keras.layers.ZeroPadding2D(padding=1)(img_input)
    x = keras.layers.Conv2D(64, 3, strides=1, use_bias=False, kernel_initializer='glorot_normal')(x)
    x = keras.layers.BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9)(x)
    x = keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = stack_fn(x)
    x = keras.layers.Flatten()(x)  # Output: (None, 25088)
    return keras.models.Model(img_input, x, name='ResNet34')

