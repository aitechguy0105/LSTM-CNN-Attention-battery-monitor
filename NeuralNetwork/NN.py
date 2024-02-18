from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model

def loadModel(input_shape, capacity_output_shape=2, cycle_output_shape=1):
    inputs = Input(shape=input_shape)

    lstm_branch = LSTM(64, activation='relu', return_sequences=True)(inputs)
    lstm_branch = Dropout(0.1)(lstm_branch)

    conv_branch = Conv1D(64, kernel_size=3, activation='relu')(lstm_branch)
    conv_branch = MaxPooling1D(pool_size=2)(conv_branch)
    conv_branch = Dropout(0.1)(conv_branch)
    conv_branch = Flatten()(conv_branch)

    dense_layer = Dense(32, activation='relu')(conv_branch)
    dense_layer = Dropout(0.1)(dense_layer)

    output_head1 = Dense(capacity_output_shape, activation='relu', name='output_head1')(dense_layer)
    output_head2 = Dense(cycle_output_shape, activation='sigmoid', name='output_head2')(dense_layer)

    model = Model(inputs=inputs, outputs=[output_head1, output_head2])

    RMSprop_optimizer = RMSprop(learning_rate=0.0001)
    model.compile(optimizer=RMSprop_optimizer, loss='mean_squared_error', metrics=['mse'])

    model.summary()

    return model