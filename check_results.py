import csv
import json
import numpy as np

run_names = [
    'Batch32_NoEstimations__t=12', 'Batch32_NoEstimations__t=0', 'Batch32_NoEstimations__t=8', 'Batch32_NoEstimations__t=11', 'Batch32_NoEstimations__t=6'
]

DATA_DIR = '/media/mike/Elements/data'

# Read CSV files with objects and labels tabulated
object_to_modulus = {}
object_to_shape = {}
object_to_material = {}
csv_file_path = f'{DATA_DIR}/objects_and_labels.csv'
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) # Skip title row
    for row in csv_reader:
        if row[14] != '':
            object_to_modulus[row[1]] = float(row[14])
            object_to_shape[row[1]] = row[2]
            object_to_material[row[1]] = row[3]

shape_log_acc = { object_to_shape[obj]:[0, 0] for obj in object_to_shape.keys() }
shape_log_acc_std = { object_to_shape[obj]:[] for obj in object_to_shape.keys() }
shape_log_diff = { object_to_shape[obj]:[0, 0] for obj in object_to_shape.keys() }
material_log_acc = { object_to_material[obj]:[0, 0] for obj in object_to_material.keys() }
material_log_acc_std = { object_to_material[obj]:[] for obj in object_to_material.keys() }
material_log_diff = { object_to_material[obj]:[0, 0] for obj in object_to_material.keys() }
poorly_predicted_pct = { obj:[0, 0] for obj in object_to_shape.keys() } # Over a factor of 100 off

for run_name in run_names:
    with open(f'./model/{run_name}/train_object_performance.json', 'r') as file:
        train_object_performance = json.load(file)
    with open(f'./model/{run_name}/val_object_performance.json', 'r') as file:
        val_object_performance = json.load(file)

    for obj in train_object_performance.keys():
        poorly_predicted_pct[obj][0] += train_object_performance[obj]['total_poorly_predicted']
        poorly_predicted_pct[obj][1] += train_object_performance[obj]['count']

    for obj in val_object_performance.keys():
        shape_log_acc[object_to_shape[obj]][0] += val_object_performance[obj]['total_log_acc']
        shape_log_acc[object_to_shape[obj]][1] += val_object_performance[obj]['count']
        shape_log_acc_std[object_to_shape[obj]].append(val_object_performance[obj]['total_log_acc'] / val_object_performance[obj]['count'])
        shape_log_diff[object_to_shape[obj]][0] += val_object_performance[obj]['total_log_diff']
        shape_log_diff[object_to_shape[obj]][1] += val_object_performance[obj]['count']

        material_log_acc[object_to_material[obj]][0] += val_object_performance[obj]['total_log_acc']
        material_log_acc[object_to_material[obj]][1] += val_object_performance[obj]['count']
        material_log_acc_std[object_to_material[obj]].append(val_object_performance[obj]['total_log_acc'] / val_object_performance[obj]['count'])
        material_log_diff[object_to_material[obj]][0] += val_object_performance[obj]['total_log_diff']
        material_log_diff[object_to_material[obj]][1] += val_object_performance[obj]['count']

        poorly_predicted_pct[obj][0] += val_object_performance[obj]['total_poorly_predicted']
        poorly_predicted_pct[obj][1] += val_object_performance[obj]['count']

for key in shape_log_acc.keys():
    if shape_log_acc[key][1] > 0:
        shape_log_acc[key] = shape_log_acc[key][0] / shape_log_acc[key][1]
    else:
        shape_log_acc[key] = 0
    if shape_log_diff[key][1] > 0:
        shape_log_diff[key] = shape_log_diff[key][0] / shape_log_diff[key][1]
    else:
        shape_log_diff[key] = 0

    shape_log_acc_std[key] = np.std(np.array(shape_log_acc_std[key]))

for key in material_log_acc.keys():
    if material_log_acc[key][1] > 0:
        material_log_acc[key] = material_log_acc[key][0] / material_log_acc[key][1]
    else:
        material_log_acc[key] = 0
    if material_log_diff[key][1] > 0:
        material_log_diff[key] = material_log_diff[key][0] / material_log_diff[key][1]
    else:
        material_log_diff[key] = 0

    material_log_acc_std[key] = np.std(np.array(material_log_acc_std[key]))

most_poorly_predicted = {}
for key in poorly_predicted_pct.keys():
    if poorly_predicted_pct[key][1] > 0:
        poorly_predicted_pct[key] = poorly_predicted_pct[key][0] / poorly_predicted_pct[key][1] # Over a factor of 100 off
    else:
        poorly_predicted_pct[key] = 0

    # Save the worst ones for display
    if poorly_predicted_pct[key] > 0.15:
        most_poorly_predicted[key] = poorly_predicted_pct[key]

print('shape_log_acc', shape_log_acc)
print('\n')
print('shape_log_acc_std', shape_log_acc_std)
print('\n')
print('shape_log_diff', shape_log_diff)
print('\n')
print('material_log_acc', material_log_acc)
print('\n')
print('material_log_acc_std', material_log_acc_std)
print('\n')
print('material_log_diff', material_log_diff)
print('\n')
print('poorly_predicted %', poorly_predicted_pct)
print('\n') 
print('most_poorly_predicted %', most_poorly_predicted)
print('\n') 