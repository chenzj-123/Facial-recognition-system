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

# ============================================================
# ArcFace Model Loader (ResNet-34 backbone)
# ============================================================

def loadModel(url='https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5'):
    """
    Loads the ArcFace (ResNet-34) embedding model and its pre-trained weights.
    If the weight file is not available locally, it will be downloaded automatically.

    Parameters
    ----------
    url : str, optional
        URL to download pre-trained ArcFace weights (default: DeepFace mirror)

    Returns
    -------
    keras.Model
        A Keras model that takes a 112×112×3 face image and outputs a 512-D embedding.
    """

    # Build the ResNet-34 base network (without classification head)
    base_model = ResNet34()
    inputs = base_model.inputs[0]
    x = base_model.outputs[0]  # Output shape: (None, 25088)

    # Add final dense and batch-norm layers for 512-D ArcFace embedding
    x = keras.layers.Dense(
        512,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_normal"
    )(x)
    embedding = keras.layers.BatchNormalization(
        momentum=0.9,
        epsilon=2e-5,
        name="embedding",
        scale=True
    )(x)

    model = keras.models.Model(inputs, embedding, name='ResNet34')

    # Download weights if not present locally
    weights_path = "arcface_weights.h5"
    if not os.path.isfile(weights_path):
        print("Downloading arcface_weights.h5...")
        gdown.download(url, weights_path, quiet=False)

    # Load weights safely (skip mismatched layers to avoid conflicts)
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


# ============================================================
# Residual Building Blocks (ResNet-34)
# ============================================================

def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """
    Defines a single residual block for ResNet-34.

    Parameters
    ----------
    x : tensor
        Input tensor.
    filters : int
        Number of filters for the convolutional layers.
    kernel_size : int, optional
        Kernel size of the main conv layers (default: 3).
    stride : int, optional
        Stride for downsampling (default: 1).
    conv_shortcut : bool, optional
        Whether to use a convolutional shortcut connection (default: True).
    name : str
        Name prefix for layers in this block.

    Returns
    -------
    tensor
        Output tensor after applying the residual block.
    """
    bn_axis = 3  # Channel axis for BatchNorm (NHWC format)

    # ----- Shortcut branch -----
    if conv_shortcut:
        shortcut = tensorflow.keras.layers.Conv2D(
            filters, 1, strides=stride, use_bias=False,
            kernel_initializer='glorot_normal', name=name + '_0_conv'
        )(x)
        shortcut = tensorflow.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_0_bn'
        )(shortcut)
    else:
        shortcut = x

    # ----- Main branch -----
    x = tensorflow.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_1_bn'
    )(x)
    x = tensorflow.keras.layers.ZeroPadding2D(padding=1, name=name + '_1_pad')(x)
    x = tensorflow.keras.layers.Conv2D(
        filters, 3, strides=1, use_bias=False,
        kernel_initializer='glorot_normal', name=name + '_1_conv'
    )(x)
    x = tensorflow.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_2_bn'
    )(x)
    x = tensorflow.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_1_prelu')(x)

    x = tensorflow.keras.layers.ZeroPadding2D(padding=1, name=name + '_2_pad')(x)
    x = tensorflow.keras.layers.Conv2D(
        filters, kernel_size, strides=stride, use_bias=False,
        kernel_initializer='glorot_normal', name=name + '_2_conv'
    )(x)
    x = tensorflow.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_3_bn'
    )(x)

    # ----- Add shortcut and main path -----
    x = tensorflow.keras.layers.Add(name=name + '_add')([shortcut, x])
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """
    Creates a stack of residual blocks for one ResNet stage.

    Parameters
    ----------
    x : tensor
        Input tensor.
    filters : int
        Number of convolutional filters.
    blocks : int
        Number of residual blocks in this stack.
    stride1 : int, optional
        Stride of the first block (controls downsampling).
    name : str
        Name prefix for this stack.

    Returns
    -------
    tensor
        Output tensor after applying the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def stack_fn(x):
    """Defines the four main residual stages for ResNet-34."""
    x = stack1(x, 64, 3, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 6, name='conv4')
    return stack1(x, 512, 3, name='conv5')


# ============================================================
# ResNet-34 Backbone
# ============================================================

def ResNet34():
    """
    Builds the ResNet-34 backbone for ArcFace.

    Input shape : (112, 112, 3)
    Output shape: (None, 25088) before projection to 512-D embedding.
    """
    img_input = keras.layers.Input(shape=(112, 112, 3))

    # Initial convolutional layer
    x = keras.layers.ZeroPadding2D(padding=1)(img_input)
    x = keras.layers.Conv2D(
        64, 3, strides=1, use_bias=False, kernel_initializer='glorot_normal'
    )(x)
    x = keras.layers.BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9)(x)
    x = keras.layers.PReLU(shared_axes=[1, 2])(x)

    # Residual stacks
    x = stack_fn(x)

    # Flatten to 1-D vector for dense projection
    x = keras.layers.Flatten()(x)

    return keras.models.Model(img_input, x, name='ResNet34')
