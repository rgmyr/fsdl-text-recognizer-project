import pathlib
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel


def all_conv_net(
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        window_width: float=16,
        window_stride: float=8) -> KerasModel:
    image_height, image_width = input_shape
    output_length, conv_dim = output_shape

    model = Sequential()
    model.add(Reshape((image_height, image_width, 1), input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # So far, this is the same as LeNet. At this point, LeNet would flatten and Dense 128.
    # Instead, we are going to use a Conv2D to slide over these outputs with window_width and window_stride,
    # and output softmax activations of shape (output_length, num_classes)./
    # In your calculation of the necessary filter size,
    # remember that padding is set to 'valid' (by default) in the network above.

    ##### Your code below (Lab 2)
    new_h = image_height // 2 - 2
    new_w = image_width // 2 - 2
    new_window_w = window_width // 2 - 2
    new_window_stride = window_stride // 2
    
    model.add(Conv2D(conv_dim, (new_h, new_window_w), (1, new_window_stride), activation='relu'))
    model.add(Dropout(0.2))
    
    num_windows = int((new_w - new_window_w) / new_window_stride) + 1
    
    model.add(Reshape((num_windows, conv_dim)))

    return model

