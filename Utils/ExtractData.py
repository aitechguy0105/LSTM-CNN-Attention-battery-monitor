import h5py
import numpy as np
import pickle
import os

def get_cell_count(f):
  return f['batch']['summary'].shape[0]

def read_cell(f, index):

  batch = f['batch']

  # fast charging policy
  policy = np.array(f[batch['policy_readable'][index, 0]]).tobytes()[::2].decode()

  # lifetime in number of cycles
  cycle_life = np.array(f[batch['cycle_life'][index, 0]])

  # voltage linearly interpolated
  vdlin = np.array(f[f['batch']['Vdlin'][index, 0]])

  # per cycle summary of measurements
  summary = {}
  summary_struct = f[batch['summary'][index,0]]
  for field in summary_struct.keys():
    summary[field] = np.hstack(summary_struct[field][0,:].tolist())

  # somehow i cant decode channel_id and barcode

  # cycles is array of dict
  # each dict has info about various fields of cycles
  cycles = []
  cycles_struct = f[f['batch']['cycles'][index, 0]]
  num_cycles = cycles_struct['I'].shape[0]

  for j in range(num_cycles):
    cycle_info = {}
    for field in cycles_struct.keys():
      cycle_info[field] = np.hstack((f[cycles_struct[field][j, 0]][()]))
    cycles.append(cycle_info)

  # Qc, Qd, (in cycles) are in Ah units

  # QCharge, QDischarge (in summary) are in Ah units

  # IR (in summary) is internal resistance

  # I (in cycles) unit is C (not A)
  # The cells are rated at 1.1 Ah
  # so 1c = 1.1A
  # 4c = 4*1.1A
  # discharging at 4c means pulling 4*1.1A out of the cell
  # charging at 5c means pumping 4*1.1A current into the cell

  # Charging policy is in format C1(Q)-C2
  # c1 and c2 are constant current
  # Q is the state of charge threshold at which current is c2
  # after c2 at state of charge 80% cell is charged at 1C or 1.1A

  # if initial cell voltage is lower than 2V
  # cell is charged at some low current (0.1C or something similar)
  # until it reaches 2v

  # once the cell reaches 3.6V cell is charged at constant voltage
  # during which voltage stays same but current drops

  return {
      'policy': policy,
      'cycle_life': cycle_life,
      'summary': summary,
      'vdlin': vdlin,
      'cycles': cycles,
  }

def runExtraction(dataDir):
  batch = {}
  batch_files = [os.path.join(dataDir, i) for i in ["batch1.mat", "batch2.mat", "batch3.mat"]]

  corrupted_cells = [
    "b1c8", "b1c10", "b1c12", "b1c13", "b1c22",
    "b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43",
  ]

  for i, batch_file in enumerate(batch_files):
    with h5py.File(batch_file, "r") as f:
      cell_count = get_cell_count(f)
      for j in range(cell_count):
        key = f"b{i+1}c{j}"
        if key not in corrupted_cells:
          print(f"reading {key}")
          batch[key] = read_cell(f, j)

  batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
  batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
  add_len = [662, 981, 1060, 208, 482]

  for i, batch1_key in enumerate(batch1_keys):

      # cycle life
      batch[batch1_key]['cycle_life'] = batch[batch1_key]['cycle_life'] + add_len[i]

      # summary
      for j in batch[batch1_key]['summary'].keys():
          # cycle in summary
          if j == 'cycle':
              batch[batch1_key]['summary'][j] = np.hstack((
                batch[batch1_key]['summary'][j],
                batch[batch2_keys[i]]['summary'][j] +
                len(batch[batch1_key]['summary'][j])
              ))
          # other things in summary
          else:
              batch[batch1_key]['summary'][j] = np.hstack((
                  batch[batch1_key]['summary'][j],
                  batch[batch2_keys[i]]['summary'][j]
              ))

      # cycles
      last_cycle = len(batch[batch1_key]['cycles'])
      for j in range(len(batch[batch2_keys[i]]['cycles'])):
          batch[batch1_key]['cycles'].append(batch[batch2_keys[i]]['cycles'][j])

      del batch[batch2_keys[i]]

  for k in batch.keys():
    with open(f"./Data/cells/{k}.pkl", "wb") as f:
      pickle.dump(batch[k], f)