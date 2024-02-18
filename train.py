import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.ExtractData import runExtraction
from Utils.PrepareData import runPrepareData
from NeuralNetwork.NN import loadModel
from tensorflow.keras import Model

from NeuralNetwork.ODENN import ode_fn, create_neural_ode_model
import sys
from tqdm.keras import TqdmCallback
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm  # Import tqdm for the progress bar



# ============================== Parameters ==============================
# 2159 cycles: dict_keys(['I', 'Qc', 'Qd', 'Qdlin', 'T', 'Tdlin', 'V', 'discharge_dQdV', 't'])
# Keys of summary: dict_keys(['IR', 'QCharge', 'QDischarge', 'Tavg', 'Tmax', 'Tmin', 'chargetime', 'cycle'])
# Data keys: ['IR' 'Tavg' 'Tmax' 'Tmin' 'chargetime' 'QCharge' 'QDischarge' 'cycle'] + Qinitial # cycle is in raw form
# Label keys: ['QCharge', 'QDischarge', 'cycle'] # cycle is converted to SOH

# ============================== Secondary testing data ==============================
# ['b3c0.pkl' 'b3c1.pkl' 'b3c3.pkl' 'b3c4.pkl' 'b3c5.pkl' 'b3c6.pkl'
#  'b3c7.pkl' 'b3c8.pkl' 'b3c9.pkl' 'b3c10.pkl' 'b3c11.pkl' 'b3c12.pkl'
#  'b3c13.pkl' 'b3c14.pkl' 'b3c15.pkl' 'b3c16.pkl' 'b3c17.pkl' 'b3c18.pkl'
#  'b3c19.pkl' 'b3c20.pkl' 'b3c21.pkl' 'b3c22.pkl' 'b3c24.pkl' 'b3c25.pkl'
#  'b3c26.pkl' 'b3c27.pkl' 'b3c28.pkl' 'b3c29.pkl' 'b3c30.pkl' 'b3c31.pkl'
#  'b3c33.pkl' 'b3c34.pkl' 'b3c35.pkl' 'b3c36.pkl' 'b3c38.pkl' 'b3c39.pkl'
#  'b3c40.pkl' 'b3c41.pkl' 'b3c44.pkl' 'b3c45.pkl']

# SOC = ((QCharge - QDischarge)/QCharge)*100
# SOH = 1 - cycle

dataDir = './Data'
cellDataDir = './Data/cells'
weightsDir_ML = './Weights/model_1.h5'
weightsDir_ODE = './Weights/model_ODE.h5'
SOC_resultsDir = './Images/SOC_SOH_results/SOC/'
SOH_resultsDir = './Images/SOC_SOH_results/SOH/'
input_keys = ['IR', 'QCharge', 'QDischarge', 'Tavg', 'Tmax', 'Tmin', 'chargetime', 'cycle']
history_input_keys = ['QCharge', 'QDischarge', 'cycle']
output_keys = ['QCharge', 'QDischarge', 'cycle']
lr = 1e-3
num_his = 1 
inputLength = 11
# epochs = 50
epochs = 5
batch_size = 16
SOH_output_shape = 1 
SOC_output_shape = 1
initial_time = 0.0
lambda_ = 0.2 # should be in (0.01, 0.4)
solution_times = np.array([1.0])
testMode = True
trainMode = True
modelType = 'ODE' # ML, ODE


# ============================== Data preparation ==============================
if (os.listdir(cellDataDir)==[]):
    runExtraction(dataDir)
    print("\n" + "-"*20 + "Step 1: Extract data to cell data" + "-"*20 + "\n")

print("\n" + "-"*20 + "Step 2: Data preparation" + "-"*20 + "\n")

x_train, y_train, train_idx, initial_SOC_train, x_test, y_test, test_idx, initial_SOC_test, x_secondary_test, y_secondary_test, secondary_test_idx, initial_SOC_secondary_test = runPrepareData(cellDataDir, input_keys, history_input_keys, output_keys, num_his)

def CreateSOC(capacity_data, initial_SOC_list, idx_length, total_capacity=1.1): # scaled in (0, 1)
    SOC = []
    start = 0
    initial_SOC = 0.
    for idx, initial_SOC in enumerate(initial_SOC_list):
        
        length = idx_length[idx]
        charge = capacity_data[:, 0][start: start+length]
        discharge = capacity_data[:, 1][start: start+length]
        start += length

        for i in range(len(discharge)):
            SOC.append(initial_SOC)
            initial_SOC += ((charge[i]-discharge[i])/total_capacity)*100.

    return np.expand_dims(np.array(SOC)/100., -1)

def ScaleCycle(cycleData, idx_length):
    start = 0
    scaledData = []
    for i in idx_length:
        eachData = cycleData[start: start+i]
        cycle_min = np.min(eachData)
        cycle_max = np.max(eachData)
        scaledData.append((eachData - cycle_min) / (cycle_max - cycle_min))
        start += i
    return np.concatenate((scaledData), axis=0)

# train + test dataset
initial_SOC_train_test = np.concatenate((initial_SOC_train, initial_SOC_test), 0)
train_test_idx = np.concatenate((train_idx, test_idx), 0)
x_train_test = np.concatenate((x_train, x_test))
y_train_test = np.concatenate((y_train, y_test))

y_train_test_capacity = y_train_test[:, :2]

y_train_test_SOC = CreateSOC(y_train_test_capacity, initial_SOC_train_test, train_test_idx)
y_train_test_SOC = np.minimum(1, np.maximum(0, y_train_test_SOC))

y_train_test_cycle = ScaleCycle(y_train_test[:, 2], train_test_idx)
y_train_test_SOH = np.expand_dims(1. - y_train_test_cycle, -1)
y_train_test_SOH_SOC = np.concatenate((y_train_test_SOH, y_train_test_SOC), axis=1)

# secondary test dataset
x_secondary_test = np.expand_dims(x_secondary_test, 1)

y_secondary_test_capacity = y_secondary_test[:, :2]
y_secondary_test_SOC = CreateSOC(y_secondary_test_capacity, initial_SOC_secondary_test, secondary_test_idx)
y_secondary_test_SOC = np.minimum(1, np.maximum(0, y_secondary_test_SOC))

y_secondary_test_cycle = ScaleCycle(y_secondary_test[:, 2], secondary_test_idx)
y_secondary_test_SOH = np.expand_dims(1. - y_secondary_test_cycle, -1)

# ============================== Training model ==============================
if trainMode:
    print("\n" + "-"*20 + "Step 3: Training NN model" + "-"*20 + "\n")

    if modelType == 'ML':
        model = loadModel((inputLength, 1), 2, 1)
        if os.path.exists(weightsDir_ML):
            model.load_weights(weightsDir_ML)

        history = model.fit(
            x_train_test, [y_train_test_capacity, y_train_test_cycle],
            epochs=epochs,
            batch_size=batch_size,
            verbose=0, callbacks=[TqdmCallback(verbose=2)],
            validation_data=(x_secondary_test, [y_secondary_test_capacity, y_secondary_test_cycle]))

        model.save(weightsDir_ML)
    else:
        model = create_neural_ode_model(inputLength, SOH_output_shape, SOC_output_shape)

        optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr)
        model.summary()
        if os.path.exists(weightsDir_ODE):
            model.load_weights(weightsDir_ODE)
        solver = tfp.math.ode.DormandPrince()

        all_epoch_loss = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            loss_count = 0.0
            for i in tqdm(range(0, y_train_test_SOH.shape[0], batch_size), desc=f"Epoch: {epoch}"):

                batch_indices = range(i, min(i+batch_size, y_train_test_SOH.shape[0]))
                selected_x_train_test = np.squeeze(tf.gather(x_train_test, indices=[1, 2, 6, 7], axis=1))
                batch_data = tf.concat([selected_x_train_test[batch_indices], y_train_test_SOH_SOC[batch_indices]], -1)

                input_tensor = tf.constant(np.expand_dims(x_train_test[batch_indices], 1), dtype=tf.float32)

                with tf.GradientTape() as tape:
                    tape.watch(model.trainable_variables)

                    solver_result = solver.solve(ode_fn, 
                                                initial_time, 
                                                batch_data, 
                                                solution_times)
                    solver_states = tf.cast(tf.squeeze(solver_result.states, axis=0)[:, -2:], dtype=tf.float32)
                    
                    predicted_Neural = model(input_tensor)

                    ############################# loss function #############################
                    ODE_loss = tf.reduce_mean(tf.square(solver_states - predicted_Neural))
                    NN_loss = tf.reduce_mean(tf.square(y_train_test_SOH_SOC[batch_indices] - predicted_Neural)) 
                    total_loss = lambda_*ODE_loss + (1.-lambda_)*NN_loss
                    ########################################################################

                    neural_gradients = tape.gradient(total_loss, model.trainable_variables)
                
                optimizer.apply_gradients(zip(neural_gradients, model.trainable_variables))

                batch_loss = total_loss.numpy()
                epoch_loss += batch_loss
                loss_count += 1.0

            avg_epoch_loss = epoch_loss / (loss_count + 1e-16)
            print(f'Epoch {epoch}, Average Loss: {avg_epoch_loss:.6f}')

            # if not all_epoch_loss or avg_epoch_loss < all_epoch_loss[-1]:
            all_epoch_loss.append(avg_epoch_loss)
            # else:
            #     break

            model.save(weightsDir_ODE)

if testMode:
    print("\n" + "-"*20 + "Step 4: Testing NN model" + "-"*20 + "\n")

    if modelType == 'ML':
        model = loadModel((inputLength, 1), 2, 1)
        model.load_weights(weightsDir_ML)

        y_secondary_test_capacity_prediction, y_secondary_test_cycle_prediction = model.predict(x_secondary_test)
        y_secondary_test_SOC_prediction = ( (y_secondary_test_capacity_prediction[:, 0] - y_secondary_test_capacity_prediction[:, 1])/y_secondary_test_capacity_prediction[:, 0] )*100 # SOC = ((QCharge - QDischarge)/QCharge)*100
        y_secondary_test_SOH_prediction = ( (1 - y_secondary_test_cycle_prediction)/1 ) * 100
    else:
        model = tf.keras.models.load_model(weightsDir_ODE)
        predicted_states = tf.squeeze(model.predict(x_secondary_test))
        y_secondary_test_SOH_prediction, y_secondary_test_SOC_prediction = predicted_states[:, 0], predicted_states[:, 1]

    last_step = 0
    next_step = 0
    y_secondary_test_SOH_prediction = np.squeeze(y_secondary_test_SOH_prediction)
    y_secondary_test_SOC_prediction = np.squeeze(y_secondary_test_SOC_prediction)

    for order, i in enumerate(secondary_test_idx):
        next_step += i
        plt.plot(y_secondary_test_SOC_prediction[last_step: next_step], c='red', label='Predicting SOC')
        plt.plot(y_secondary_test_SOC[last_step: next_step], c='black', label='True SOC')
        plt.title(f'Data {order}: SOC')
        plt.legend()
        plt.savefig(os.path.join(SOC_resultsDir ,f'SOC_{order}.png'), facecolor='white')
        # plt.show()
        plt.close()

        plt.plot(y_secondary_test_SOH_prediction[last_step: next_step], c='red', label='Predicting SOH')
        plt.plot(y_secondary_test_SOH[last_step: next_step], c='black', label='True SOH')
        plt.title(f'Data {order} SOH')
        plt.legend()
        plt.savefig(os.path.join(SOH_resultsDir ,f'SOH_{order}.png'), facecolor='white')
        # plt.show()
        plt.close()

        last_step += i


