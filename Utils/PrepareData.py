import numpy as np
import pickle
import os
import re
from scipy import signal

def get_cell_data(cell, DATA_DIR, apply_filter=False):
  cell_data = pickle.load(open(os.path.join(DATA_DIR, cell), "rb"))
  if apply_filter:
    cell_data['summary']['QCharge'] = signal.medfilt(cell_data['summary']['QCharge'])
    cell_data['summary']['QDischarge'] = signal.medfilt(cell_data['summary']['QDischarge'])
  return cell_data

# sort files by batch and cell number
def sort_key(f):
  match = re.match(r"^b(\d+)c(\d+)", f)
  batch, cell = int(match[1]), int(match[2])
  return batch * 1000 + cell

def gatherData(data):
  gatheredData = np.array([])
  for key in data['summary'].keys():
    if gatheredData.any() == False:
      gatheredData = np.expand_dims(data['summary'][key], 1)
    else:
      gatheredData = np.concatenate(( gatheredData, np.expand_dims(data['summary'][key], 1) ), 1)

  # ################## Min Max Scale for cycles #################################
  # cycle_min = np.min(gatheredData[:, -1])
  # cycle_max = np.max(gatheredData[:, -1])
  # gatheredData[:, -1] = (gatheredData[:, -1] - cycle_min) / (cycle_max - cycle_min)
  # #############################################################################

  return np.array(gatheredData)

def createDataLabels(totalData, name, input_keys, history_input_keys, output_keys, num_his=2):
  finalData = np.array([])
  finalLabel = np.array([])

  nameDataPoint = []
  nameHisDataPoint = []
  nameLabelPoint = []

  initial_SOC_list = []
  for data in totalData:
    policy_data = data['policy']
    SoC_init = float(policy_data.split('(')[1].split(')')[0][:-1])
    initial_SOC_list.append(SoC_init) #[init_SOC, cell_length]

    gatheredData = gatherData(data)
    keyData = np.array([])
    keyHisData = np.array([])
    keyLabel = np.array([])

    # loop of keys in data
    for key_idx, key in enumerate(data['summary'].keys()):
      keyDataPoint = []
      keyHisDataPoint = []
      keyLabelPoint = []

      # get data column according to 
      for i in range(1+num_his, gatheredData.shape[0]):
        input = [gatheredData[i, key_idx]]
        hisInput = gatheredData[i-num_his: i, key_idx]

        if key in input_keys:
          keyDataPoint.append(input)
          nameDataPoint.append(key) if key not in nameDataPoint else None

        if key in history_input_keys:
          keyHisDataPoint.append(hisInput)
          nameHisDataPoint.append(key) if key not in nameHisDataPoint else None

        if key in output_keys:
          keyLabelPoint.append(input)
          nameLabelPoint.append(key) if key not in nameLabelPoint else None

      if np.array(keyDataPoint).any() == True:
        if np.array(keyData).any() == False:
          keyData = np.array(keyDataPoint)
        else:
          keyData = np.concatenate(( keyData, np.array(keyDataPoint) ), -1)

      if np.array(keyHisDataPoint).any() == True:
        if np.array(keyHisData).any() == False:
          keyHisData = np.array(keyHisDataPoint)
        else:
          keyHisData = np.concatenate(( keyHisData, np.array(keyHisDataPoint) ), -1)

      if np.array(keyLabelPoint).any() == True:
        if np.array(keyLabel).any() == False:
          keyLabel = np.array(keyLabelPoint)
        else:
          keyLabel = np.concatenate(( keyLabel, np.array(keyLabelPoint) ), -1)

    # concatenate key's data, label together
    if np.array(finalData).any() == False and np.array(finalLabel).any() == False:
      finalData = np.concatenate((keyData, keyHisData), 1)
      finalLabel = keyLabel
    else:
      conData = np.concatenate((keyData, keyHisData), 1)
      finalData = np.concatenate((finalData, conData), 0)
      finalLabel = np.concatenate((finalLabel, keyLabel), 0)

  finalData = np.expand_dims(finalData, -1)

  print(f'\n{name} set: ')
  print(f'Data shape: {finalData.shape}')
  print(f'Data keys: {np.concatenate((nameDataPoint, nameHisDataPoint))}')
  print(f'Label shape: {finalLabel.shape}')
  print(f'Label keys: {nameLabelPoint}\n')
  return finalData, finalLabel, np.array(initial_SOC_list)



def getIdxData(data, num_his):
    idx = []
    for i in range(len(data)):
        idx.append(data[i]['summary']['QCharge'].shape[0])

    idx = np.array(idx) - num_his - 1
    return idx



def runPrepareData(cellDataDir, input_keys, history_input_keys, output_keys, num_his):
    files = os.listdir(cellDataDir)
    files.sort(key=sort_key)
    files = np.array(files)

    # split data in the same way as paper
    numBat1 = len([f for f in files if f.startswith("b1")])
    numBat2 = len([f for f in files if f.startswith("b2")])
    numBat3 = len([f for f in files if f.startswith("b3")])
    numBat = numBat1 + numBat2 + numBat3

    test_ind = np.hstack((np.arange(0, (numBat1+numBat2), 2), 83))
    train_ind = np.arange(1, (numBat1+numBat2 - 1), 2)
    secondary_test_ind = np.arange(numBat-numBat3, numBat) # the authors acquired this data after model developement

    print(f'Number of cells in testing set: {len(test_ind)}\n')
    print(f'Number of cells in training set: {len(train_ind)}\n')
    print(f'Number of cells in validation set: {len(secondary_test_ind)}\n')

    defective_cells = [
        (3, 2237, '3.6C(80%)-3.6C'),
        (7, 461, '4.8C(80%)-4.8C'),
        (21, 489, '4.65C(19%)-4.85C'),
        (15, 511, '3.6C(30%)-6C'),
        (16, 561, '3.6C(9%)-5C'),
        (10, 148, '2C(10%)-6C'),
    ]

    defective_cells_found = []

    for i, f in enumerate(files):
        cell_data = get_cell_data(f, cellDataDir)
        cycle_life = cell_data['cycle_life'][0][0]
        policy = cell_data['policy']
        for cell in defective_cells:
            if cell[1] == cycle_life and cell[2] == policy:
                defective_cells_found.append((cell[0], i, f))

    # remove 7, 21, 15, 16, 10 from train_ind, test_ind
    remove_channels = [7, 21, 15, 16, 10]
    remove_ind = [c[1] for c in defective_cells_found if c[0] in remove_channels]

    train_ind = np.array([ind for ind in train_ind if ind not in remove_ind])
    test_ind = np.array([ind for ind in test_ind if ind not in remove_ind])
    secondary_test_ind = np.array([ind for ind in secondary_test_ind if ind not in remove_ind])

    train_data = [get_cell_data(i, cellDataDir) for i in files[train_ind]]
    test_data = [get_cell_data(i, cellDataDir) for i in files[test_ind]]
    secondary_test_data = [get_cell_data(i, cellDataDir) for i in files[secondary_test_ind]]

    cell_data = train_data[0]
    print(f"{len(cell_data['cycles'])} cycles: {cell_data['cycles'][0].keys()}")
    print(f"\nKeys of summary: {cell_data['summary'].keys()}")

    ################ The measure the length of each cell, using to seperate secondary test data ################ 
    train_idx = getIdxData(train_data, num_his)
    test_idx = getIdxData(test_data, num_his)
    secondary_test_idx = getIdxData(secondary_test_data, num_his)
    ############################################################################################################

    x_train, y_train, initial_SOC_train = createDataLabels(train_data, "Train", input_keys, history_input_keys, output_keys, num_his=num_his)
    x_test, y_test, initial_SOC_test = createDataLabels(test_data, "Test", input_keys, history_input_keys, output_keys, num_his=num_his)
    x_secondary_test, y_secondary_test, initial_SOC_secondary_test = createDataLabels(secondary_test_data, "Secondary test", input_keys, history_input_keys, output_keys, num_his=num_his)

    return x_train, y_train, train_idx, initial_SOC_train, x_test, y_test, test_idx, initial_SOC_test, x_secondary_test, y_secondary_test, secondary_test_idx, initial_SOC_secondary_test