from keras import backend as K
from keras.layers import Conv1D, Dense, Flatten, Dropout,MaxPooling1D, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers import Layer

def _bn_relu(layer, dropout=0, **params):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    if params["conv_batch_norm"]:
        layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D
    from keras.regularizers import l2
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer=params["conv_init"],
        kernel_regularizer=l2(params["conv_l2"])
    )(layer)
    return layer


def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],
                    params["conv_num_filters_start"],
                    subsample_length=subsample_length,
                    **params)
        layer = _bn_relu(layer,
                         dropout=params["conv_dropout"],
                         **params)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from keras.layers import Add
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=params["conv_dropout"] if i > 0 else 0,
                **params)
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params)

    # Add learnable Scalar
    if not params["conv_batch_norm"]:
        # res_multiplier = K.variable(params["skip_init_a"], dtype='float32', name='skipinit')
        # res_multiplier._trainable = True
        # layer = Lambda(lambda x: x * res_multiplier)(layer)
        layer = ScaleLayer(params["skip_init_a"])(layer)

    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * num_start_filters

def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    return layer

def add_output_layer(layer, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    return Activation('softmax')(layer)

def add_compile(model, **params):
    from keras.optimizers import Adam, SGD

    if params["optimizer"] == 'adam':
        optimizer = Adam(lr=params["learning_rate"])
    else:
        optimizer = SGD(
            lr=params["learning_rate"], momentum=0.9
        )

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

class ScaleLayer(Layer):
    def __init__(self, alpha=0):
      super(ScaleLayer, self).__init__()
      self.scale = K.variable(alpha, dtype='float32', name='skipinit')
      # self.scale = tf.Variable(1.)

    def call(self, inputs):
      return inputs * self.scale

def build_network(**params):
    from keras.models import Model
    from keras.layers import Input
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')
    # original
    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)
    #
    # layer = add_conv_layers(inputs, **params)

    output = add_output_layer(layer, **params)
    model = Model(inputs=[inputs], outputs=[output])
    if params.get("compile", True):
        add_compile(model, **params)

    # small
    # filter_length = params["conv_filter_length"]
    # num_filters = params["conv_num_filters_start"]
    # layer = Conv1D(
    #     filters=num_filters,
    #     kernel_size=filter_length,
    #     strides=1,
    #     padding='same',
    #     kernel_initializer=params["conv_init"])(inputs)
    # # x = Conv1D(filters=filter_length,
    # #            kernel_size=kernel_size,
    # #            padding='same',
    # #            strides=1,
    # #            kernel_initializer='he_normal')(inputs)
    # layer = Dropout(params["conv_dropout"])(layer)
    #
    #
    # # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # # x = MaxPooling1D(pool_size=2, strides=2)(x)
    # # x = Dropout(config.drop_rate)(x)
    # x = Conv1D(filters=filter_length,
    #            kernel_size=kernel_size,
    #            padding='same',
    #            strides=1,
    #            kernel_initializer='he_normal')(x)
    # # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPooling1D(pool_size=2, strides=2)(x)
    # # x = Dropout(config.drop_rate)(x)
    # # similar implementation to maxpool
    # # x = Dropout(0.2)(x)
    # # x = BatchNormalization()(x)
    # ## 2 convolutional block (conv,BN, relu)
    # x = Conv1D(filters=filter_length,
    #            kernel_size=kernel_size,
    #            padding='same',
    #            strides=1,
    #            kernel_initializer='he_normal')(x)
    # # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPooling1D(pool_size=2, strides=2)(x)
    # # x = Dropout(config.drop_rate)(x)
    # ## 3 convolutional block (conv,BN, relu)
    # x = Conv1D(filters=filter_length,
    #            kernel_size=kernel_size,
    #            padding='same',
    #            strides=1,
    #            kernel_initializer='he_normal')(x)
    # # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # # x = Dropout(config.drop_rate)(x)
    # x = MaxPooling1D(pool_size=2, strides=2)(x)
    #
    # # filter size : 32, filter length : 16
    # # w/o drop out : 0.83
    # # w dropout : 0.89
    # # after flatten : 0.86
    # # similar implementation to maxpool
    # # x = Dropout(config.drop_rate)(x)
    #
    # # filter size : 16, filter length : 7
    # # w/o drop out : 0.82
    # # w dropout : 0.89
    # # after flatten : 0.80
    # # all dropout : 0.91
    # # 1 dropout : 0.89
    # # x = Dropout(config.drop_rate)(x)
    #
    # # Final bit
    # # x = BatchNormalization()(x)
    # # x = Activation('relu')(x)
    # # x = Flatten()(x)
    # # x = Dropout(config.drop_rate)(x)
    #
    # # x = Dense(64, activation='relu')(x)
    # # x = Dropout(0.5)(x)
    #
    # # x = Dense(64, activation='relu')(x)
    # # x = Dropout(0.5)(x)
    #
    # # out = TimeDistributed(Dense(4, activation='softmax')(x))
    #
    # layer = TimeDistributed(Dense(params["num_categories"]))(x)
    # out = Activation('softmax')(layer)
    # model = Model(inputs=inputs, outputs=out)
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.summary()
    return model
