import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, GRU, Conv1D, MaxPooling1D, Flatten, Dropout, AveragePooling1D,\
                                    Input, Concatenate, Bidirectional, Lambda, LSTM, Attention, \
                                    Activation, BatchNormalization, Add, Concatenate, ConvLSTM1D

from keras import layers, regularizers
import keras.backend as K
from tensorflow.keras.models import Model

from tensorflow_addons.layers import MultiHeadAttention

def create_neural_ode_model(input_length, SOH_output_length, SOC_output_length):
    inputs = Input(shape=(1, input_length, 1))

    conv_branch = ConvLSTM1D(24, kernel_size=3, padding='valid', activation='relu')(inputs)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Dropout(0.3)(conv_branch)

    conv_branch = AveragePooling1D(pool_size=3, strides=2, padding='valid')(conv_branch)

    conv_branch = Bidirectional(GRU(24, return_sequences=False, activation='relu'))(conv_branch)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Dropout(0.3)(conv_branch)

    # Residual connection
    residual_connection = ConvLSTM1D(24, kernel_size=3, padding='valid', activation='relu')(inputs)
    residual_connection = BatchNormalization()(residual_connection)
    residual_connection = Dropout(0.3)(residual_connection)

    residual_connection = AveragePooling1D(pool_size=3, strides=2, padding='valid')(residual_connection)

    residual_connection = Bidirectional(LSTM(24, return_sequences=False, activation='relu'))(residual_connection)
    residual_connection = BatchNormalization()(residual_connection)
    residual_connection = Dropout(0.3)(residual_connection)

    # attention_branch = MultiHeadAttention(head_size=4, num_heads=2)([conv_branch, conv_branch, residual_connection])
    attention_branch = Attention()([conv_branch, residual_connection])
    attention_branch = BatchNormalization()(attention_branch)
    attention_branch = Dropout(0.3)(attention_branch)

    merged_branches = Concatenate(axis=-1)([conv_branch, residual_connection, attention_branch])
    flatten_layer = Flatten()(merged_branches)

    SOH_output = Dense(SOH_output_length, activation='sigmoid', name='SOH_output')(flatten_layer)
    SOC_output = Dense(SOC_output_length, activation='sigmoid', name='SOC_output')(flatten_layer)

    outputs = Concatenate(axis=-1)([SOH_output, SOC_output])

    model = Model(inputs=inputs, outputs=outputs)
    return model

@tf.custom_gradient
def ode_fn(t, y, Qmax=1.1):
    QCharge, QDischarge, chargetime, cycle, SOH, SOC = tf.split(y, 6, axis=-1)

    charge_rate = QCharge / (chargetime)
    discharge_rate = QDischarge / (t - chargetime)

    dSOH_dt = -0.001 * cycle
    dSOC_dt = -0.0001 * ((charge_rate - discharge_rate)/Qmax)

    dy_dt = tf.concat([QCharge, QDischarge, chargetime, cycle, dSOH_dt, dSOC_dt], axis=-1)

    @tf.custom_gradient
    def grad(dy_dt):
        # Define the gradient of the state with respect to time 
        dQCharge_dt = tf.zeros_like(QCharge)  
        dQDischarge_dt = tf.zeros_like(QDischarge)   
        dchargetime_dt = tf.zeros_like(chargetime)   
        dcycle_dt = tf.zeros_like(cycle)  
        ddSOH_dt = tf.zeros_like(dSOH_dt)   
        ddSOC_dt = tf.zeros_like(dSOC_dt)  

        # Return the gradient of the state with respect to time
        grad_state = tf.concat([dQCharge_dt, dQDischarge_dt, dchargetime_dt, dcycle_dt, ddSOH_dt, ddSOC_dt], axis=-1)

        return grad_state, None

    return dy_dt, grad

######################################### VERSION 1 #########################################
# def create_neural_ode_model(input_length, SOH_output_length, SOC_output_length):
#     inputs = Input(shape=(input_length, 1))

#     conv_branch = ConvLSTM1D(24, kernel_size=3, strides=2, padding='valid')(inputs)
#     conv_branch = BatchNormalization()(conv_branch)
#     conv_branch = Activation('relu')(conv_branch)
#     conv_branch = Dropout(0.3)(conv_branch)

#     conv_branch = AveragePooling1D(pool_size=3, strides=2, padding='valid')(conv_branch)

#     conv_branch = Bidirectional(GRU(24, return_sequences=False))(conv_branch)
#     conv_branch = BatchNormalization()(conv_branch)
#     conv_branch = Activation('relu')(conv_branch)
#     conv_branch = Dropout(0.3)(conv_branch)

#     # Residual connection
#     residual_connection = ConvLSTM1D(24, kernel_size=3, strides=2, padding='valid')(inputs)
#     residual_connection = BatchNormalization()(residual_connection)
#     residual_connection = Activation('relu')(residual_connection)
#     residual_connection = Dropout(0.3)(residual_connection)

#     residual_connection = AveragePooling1D(pool_size=3, strides=2, padding='valid')(residual_connection)

#     residual_connection = Bidirectional(LSTM(24, return_sequences=False))(residual_connection)
#     residual_connection = BatchNormalization()(residual_connection)
#     residual_connection = Activation('relu')(residual_connection)
#     residual_connection = Dropout(0.3)(residual_connection)

#     # attention_branch = MultiHeadAttention(head_size=4, num_heads=2)([conv_branch, conv_branch, residual_connection])
#     attention_branch = Attention()([conv_branch, residual_connection])
#     attention_branch = BatchNormalization()(attention_branch)
#     attention_branch = Activation('relu')(attention_branch)
#     attention_branch = Dropout(0.3)(attention_branch)

#     flatten_layer = Flatten()(attention_branch)

#     SOH_output = Dense(SOH_output_length, activation='sigmoid', name='SOH_output')(flatten_layer)
#     SOC_output = Dense(SOC_output_length, activation='sigmoid', name='SOC_output')(flatten_layer)

#     outputs = Concatenate(axis=-1)([SOH_output, SOC_output])

#     model = Model(inputs=inputs, outputs=outputs)
#     return model