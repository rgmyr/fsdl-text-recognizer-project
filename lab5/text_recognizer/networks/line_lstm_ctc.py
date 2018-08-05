from boltons.cacheutils import cachedproperty
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.all_conv import all_conv_net
from text_recognizer.networks.misc import slide_window
from text_recognizer.networks.ctc import ctc_decode


def line_lstm_ctc(input_shape, output_shape, window_width=18, window_stride=6, conv_dim=256, lstm_dim=256):
    gpu_present = len(device_lib.list_local_devices()) > 1
    lstm_fn = CuDNNLSTM if gpu_present else LSTM
    
    image_height, image_width = input_shape
    output_length, num_classes = output_shape
    
    num_windows = int((image_width - window_width) / window_stride) + 1
    if num_windows < output_length:
        raise ValueError(f'Window width/stride need to generate at least {output_length} windows (currently {num_windows})')

    image_input = Input(shape=input_shape, name='image')
    y_true = Input(shape=(output_length,), name='y_true')
    input_length = Input(shape=(1,), name='input_length')
    label_length = Input(shape=(1,), name='label_length')

    # Make a ConvNet with windowed output
    convnet = all_conv_net((image_height, image_width), conv_dim, window_width, window_stride)
    conv_out = convnet(image_input)
    # (num_windows, conv_dim)

    # 3 LSTM layers, with residual connections
    lstm_output0 = Bidirectional(lstm_fn(lstm_dim*2, return_sequences=True))(conv_out)
    
    lstm_output1 = Bidirectional(lstm_fn(lstm_dim, return_sequences=True))(lstm_output0)
    #lstm_output1 = Add()([lstm_output0, lstm_output1])
    
    lstm_output = Bidirectional(lstm_fn(lstm_dim//2, return_sequences=True))(lstm_output1)
    #lstm_output = Add()([lstm_output1, lstm_output])

    softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output)
    # (num_windows, num_classes)
    ##### Your code above (Lab 3)

    input_length_processed = Lambda(
        lambda x, num_windows=None: x * num_windows,
        arguments={'num_windows': num_windows}
    )(input_length)

    ctc_loss_output = Lambda(
        lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]),
        name='ctc_loss'
    )([y_true, softmax_output, input_length_processed, label_length])

    ctc_decoded_output = Lambda(
        lambda x: ctc_decode(x[0], x[1], output_length),
        name='ctc_decoded'
    )([softmax_output, input_length_processed])

    full_model = KerasModel(
        inputs=[image_input, y_true, input_length, label_length],
        outputs=[ctc_loss_output, ctc_decoded_output]
    )
    return full_model

