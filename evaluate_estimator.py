import os
import csv
import json
import pickle
import random
import copy
import math
import numpy as np
from tqdm import tqdm

from scipy.optimize import curve_fit
# from tactile_estimate import *

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

def random_hex_color():
    # Generate random values for red, green, and blue
    R = random.randint(0, 255)
    G = random.randint(0, 255)
    B = random.randint(0, 255)

    # Format the values as hexadecimal and concatenate them
    return "#{:02X}{:02X}{:02X}".format(R, G, B)

def closest_non_nan_element(numbers, index):
    closest_distance = float('inf')
    closest_element = None

    for i in range(len(numbers)):
        if numbers[i] > 0:
            distance = abs(i - index)
            if distance < closest_distance:
                closest_distance = distance
                closest_element = numbers[i]

    if closest_distance > 1e10: return 0

    return closest_element

DATA_DIR = './data' # '/media/mike/Elements/data'

# Objects to exclude from evaluation
EXCLUDE = [
            'playdoh', 'silly_puty', 'racquet_ball', 'blue_sponge_dry', 'blue_sponge_wet', \
            'blue_foam_brick', 'green_foam_brick', # 'yellow_foam_brick', 'red_foam_brick', 
            'apple', 'orange', 'strawberry', 'ripe_banana', 'unripe_banana', 
            'lacrosse_ball', 'scotch_brite', 'fake_washer_stack', 'cork', 'rubber_spatula',
            'baseball', 'plastic_measuring_cup', 'whiteboard_eraser', 'lifesaver_hard', 'cutting_board',
            'plastic_knife', 'plastic_fork', 'plastic_spoon', 'plastic_fork_white'
        ]

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

# Find an optimal linear scaling for a set of modulus predictions
def scale_predictions(prediction_dict, scale_cutoff=1e11, quadratic=False):
    x, y = [], []
    for object_name in prediction_dict.keys():
        if object_name in EXCLUDE: continue
        for E in prediction_dict[object_name]:
            if E > 0 and not math.isnan(E) and E < 1e7 and object_to_modulus[object_name] < scale_cutoff:
                x.append(E)
                y.append(object_to_modulus[object_name])

    # Filter out outliers for fitting
    x, y = np.array(x), np.array(y)
    outlier_mask = np.abs(x - np.mean(x)) < np.std(x)
    x_filtered, y_filtered = x[outlier_mask], y[outlier_mask]
    
    if len(x_filtered) < 10: return prediction_dict

    if quadratic:
        poly = np.polyfit(np.log2(x_filtered), np.log10(y_filtered), 2)
    else:
        poly = np.polyfit(np.log2(x_filtered), np.log10(y_filtered), 1)

    # Scale all predictions accordingly
    scaled_prediction_dict = {}
    for object_name in prediction_dict.keys():
        scaled_prediction_dict[object_name] = []
        if object_name in EXCLUDE: continue
        for E in prediction_dict[object_name]:
            if E > 0 and not math.isnan(E):
                if quadratic:
                    E_scaled = 10**((np.log2(E)**2)*poly[0] + np.log2(E)*poly[1] + poly[2])
                else:
                    E_scaled = 10**(np.log2(E)*poly[0] + poly[1])
                scaled_prediction_dict[object_name].append(E_scaled)
            else:
                scaled_prediction_dict[object_name].append(E)

    return scaled_prediction_dict

# Compute statistics to evaluate the performance of estimation method
def compute_estimation_stats(prediction_dict):
    log_diff = []
    nan_count, total_count = 0, 0
    for object_name in prediction_dict.keys():
        if object_name in EXCLUDE: continue
        for E in prediction_dict[object_name]:
            if E > 0 and not math.isnan(E):
                assert not math.isnan(E)
                log_diff.append(abs(np.log10(E) - np.log10(object_to_modulus[object_name])))
            else:
                nan_count += 1
        total_count += len(prediction_dict[object_name])
    
    log_diff = np.array(log_diff)
    log_diff = log_diff[~np.isnan(log_diff)]
    return {
        'avg_log_diff': log_diff.mean(),
        'log_accuracy': np.sum(log_diff < 1) / len(log_diff),
        'nan_pct': nan_count / total_count,
    }

# Compute log difference of predictions per object to see which object is worst
def compute_object_performance(prediction_dicts):
    object_names = list(prediction_dicts[0].keys())
    obj_prediction_log_diff = { obj:[] for obj in object_names }
    objects_avg_log_diff = { obj:0 for obj in object_names }

    for prediction_dict in prediction_dicts:
        for object_name in object_names:
            for E in prediction_dict[object_name]:
                if E > 0:
                    log_diff = np.log10(E) - np.log10(object_to_modulus[object_name])
                    obj_prediction_log_diff[object_name].append(log_diff)

    for object_name in object_names:
        if len(obj_prediction_log_diff[object_name]) > 0:
            objects_avg_log_diff[object_name] = sum(obj_prediction_log_diff[object_name]) / len(obj_prediction_log_diff[object_name])

    return obj_prediction_log_diff, objects_avg_log_diff

# Compute log difference of predictions per shape
def compute_shape_performance(prediction_dict):
    shapes = ['Rectangular', 'Sphere', 'Cylinder', 'Hex', 'Irregular']
    shape_prediction_log_diff = { shape:[] for shape in shapes }

    for object_name in object_names:
        shape = object_to_shape[object_name]
        if shape not in shapes: continue
        for E in prediction_dict[object_name]:
            if E > 0:
                log_diff = abs(np.log10(E) - np.log10(object_to_modulus[object_name]))
                shape_prediction_log_diff[shape].append(log_diff)

    shape_log_diff = { shape:0 for shape in shapes }
    shape_log_acc = { shape:0 for shape in shapes }
    for shape in shapes:
        if len(shape_prediction_log_diff[shape]) > 0:
            shape_log_diff[shape] = sum(shape_prediction_log_diff[shape]) / len(shape_prediction_log_diff[shape])
            shape_log_acc[shape] = sum(np.array(shape_prediction_log_diff[shape]) <= 1) / len(shape_prediction_log_diff[shape])

    return shape_log_diff, shape_log_acc

# Compute log difference of predictions per material
def compute_material_performance(prediction_dict):
    materials = ['Foam', 'Rubber', 'Plastic', 'Wood', 'Metal']
    mat_prediction_log_diff = { mat:[] for mat in materials }

    for object_name in object_names:
        mat = object_to_material[object_name]
        if mat not in materials: continue
        for E in prediction_dict[object_name]:
            if E > 0:
                log_diff = abs(np.log10(E) - np.log10(object_to_modulus[object_name]))
                mat_prediction_log_diff[mat].append(log_diff)

    mat_log_diff = { mat:0 for mat in materials }
    mat_log_acc = { mat:0 for mat in materials }
    for mat in materials:
        if len(mat_prediction_log_diff[mat]) > 0:
            mat_log_diff[mat] = sum(mat_prediction_log_diff[mat]) / len(mat_prediction_log_diff[mat])
            mat_log_acc[mat] = sum(np.array(mat_prediction_log_diff[mat]) <= 1) / len(mat_prediction_log_diff[mat])

    return mat_log_diff, mat_log_acc

# Compute log difference and other stats for each contact mask
def compute_contact_mask_performance(list_of_configs, list_of_stats):
    assert len(list_of_configs) == len(list_of_stats)
    contact_mask_stats = {}
    empty_stats_dict = {
        'avg_log_diff': 0,
        'log_accuracy': 0,
        'nan_pct': 0,
        'count': 0
    }

    for i in range(len(list_of_configs)):
        contact_mask = list_of_configs[i]['contact_mask']
        if contact_mask not in contact_mask_stats.keys():
            contact_mask_stats[contact_mask] = copy.deepcopy(empty_stats_dict)
        contact_mask_stats[contact_mask]['avg_log_diff'] += list_of_stats[i]['avg_log_diff']
        contact_mask_stats[contact_mask]['log_accuracy'] += list_of_stats[i]['log_accuracy']
        contact_mask_stats[contact_mask]['nan_pct']      += list_of_stats[i]['nan_pct']
        contact_mask_stats[contact_mask]['count']        += 1

    for contact_mask in contact_mask_stats.keys():
        contact_mask_stats[contact_mask]['avg_log_diff'] /= contact_mask_stats[contact_mask]['count']
        contact_mask_stats[contact_mask]['log_accuracy'] /= contact_mask_stats[contact_mask]['count']
        contact_mask_stats[contact_mask]['nan_pct']      /= contact_mask_stats[contact_mask]['count']

    return contact_mask_stats

# Plot on log scale to see how performance is
def plot_performance(plot_title, prediction_dict, label_dict):
    material_to_color = {
        'Foam': 'firebrick',
        'Plastic': 'forestgreen',
        'Wood': 'goldenrod',
        'Paper': 'yellow',
        'Glass': 'darkgray',
        'Ceramic': 'pink',
        'Rubber': 'slateblue',
        'Metal': 'royalblue',
        'Food': 'darkorange',
    }
    material_prediction_data = {
        mat : [] for mat in material_to_color.keys()
    }
    material_label_data = {
        mat : [] for mat in material_to_color.keys()
    }
    
    for obj in prediction_dict.keys():
        if len(prediction_dict[obj]) == 0: continue
        assert obj in object_to_material.keys()
        mat = object_to_material[obj]
        for E in prediction_dict[obj]:
            if E > 0 and not math.isnan(E):
                assert not math.isnan(E)
                material_prediction_data[mat].append(E)
                material_label_data[mat].append(label_dict[obj])

    # Create plot
    mpl.rcParams['font.family'] = ['serif']
    mpl.rcParams['font.serif'] = ['Times New Roman']
    plt.figure()
    plt.plot([100, 10**12], [100, 10**12], 'k--', label='_')
    plt.fill_between([100, 10**12], [10**1, 10**11], [10**3, 10**13], color='gray', alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')

    for mat in material_to_color.keys():
        plt.plot(material_label_data[mat], material_prediction_data[mat], '.', markersize=10, color=material_to_color[mat], label=mat)

    plt.xlabel("Ground Truth Modulus ($E$) [$\\frac{N}{m^2}$]", fontsize=12)
    plt.ylabel("Predicted Modulus ($\\tilde{E}$) [$\\frac{N}{m^2}$]", fontsize=12)
    plt.xlim([100, 10**12])
    plt.ylim([100, 10**12])
    plt.title(plot_title, fontsize=14)

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    plt.tick_params(axis='both', which='both', labelsize=10)

    plt.savefig(f'./figures/{plot_title.replace(" ", "_")}.png')
    plt.show()

    return


if __name__ == '__main__':

    empty_estimate_dict     = { obj:[] for obj in object_to_modulus.keys() }
    no_tactile_estimates    = { obj:[] for obj in object_to_modulus.keys() }
    naive_estimates         = []
    naive_configs           = []
    hertz_estimates         = []
    hertz_configs           = []
    MDR_estimates           = []
    MDR_configs             = []

    naive_other_estimates         = []
    naive_other_configs           = []
    hertz_other_estimates         = []
    hertz_other_configs           = []
    MDR_other_estimates           = []
    MDR_other_configs             = []

    naive_both_sides_estimates     = []
    naive_both_sides_configs       = []
    hertz_both_sides_estimates     = []
    hertz_both_sides_configs       = []
    MDR_both_sides_estimates       = []
    MDR_both_sides_configs         = []

    for object_name in tqdm(sorted(os.listdir(f'{DATA_DIR}/estimations_new_new'))):
        if object_name.count('.') > 0: continue
        if object_name in EXCLUDE: continue

        for trial_folder in os.listdir(f'{DATA_DIR}/estimations_new_new/{object_name}'):
            grasp_dir = f'{DATA_DIR}/estimations_new_new/{object_name}/{trial_folder}'




            # Unpack no tactile estimation
            with open(f'{grasp_dir}/no_tactile/0.pkl', 'rb') as file:
                E_i = pickle.load(file)
            no_tactile_estimates[object_name].append(E_i)





            # Unpack naive estimations for each config type
            i = 0
            method = 'naive'
            for contact_mask in sorted(os.listdir(f'{grasp_dir}/{method}')):
                for file_name in sorted(os.listdir(f'{grasp_dir}/{method}/{contact_mask}')):
                    if file_name.count('.pkl') == 0: continue

                    # Extract info
                    with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name}', 'rb') as file:
                        E_i = pickle.load(file)

                    if i > len(naive_estimates) - 1:
                        naive_estimates.append(copy.deepcopy(empty_estimate_dict))
                        with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name.split(".")[0]}.json', 'r') as file:
                            config_i = json.load(file)
                        naive_configs.append(config_i)
                    
                    naive_estimates[i][object_name].append(E_i)
                    i += 1

            # Unpack Hertzian estimations for each config type
            i = 0
            method = 'hertz'
            for contact_mask in sorted(os.listdir(f'{grasp_dir}/{method}')):
                for file_name in sorted(os.listdir(f'{grasp_dir}/{method}/{contact_mask}')):
                    if file_name.count('.pkl') == 0: continue

                    # Extract info
                    with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name}', 'rb') as file:
                        E_i = pickle.load(file)

                    if i > len(hertz_estimates) - 1:
                        hertz_estimates.append(copy.deepcopy(empty_estimate_dict))
                        with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name.split(".")[0]}.json', 'r') as file:
                            config_i = json.load(file)
                        hertz_configs.append(config_i)

                    hertz_estimates[i][object_name].append(E_i)
                    i += 1

            # Unpack MDR estimations for each config type
            i = 0
            method = 'MDR'
            for contact_mask in sorted(os.listdir(f'{grasp_dir}/{method}')):
                for file_name in sorted(os.listdir(f'{grasp_dir}/{method}/{contact_mask}')):
                    if file_name.count('.pkl') == 0: continue

                    # Extract info
                    with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name}', 'rb') as file:
                        E_i = pickle.load(file)

                    if i > len(MDR_estimates) - 1:
                        MDR_estimates.append(copy.deepcopy(empty_estimate_dict))
                        with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name.split(".")[0]}.json', 'r') as file:
                            config_i = json.load(file)
                        MDR_configs.append(config_i)

                    MDR_estimates[i][object_name].append(E_i)
                    i += 1

            # Unpack naive estimations for each config type
            i = 0
            method = 'naive_other'
            for contact_mask in sorted(os.listdir(f'{grasp_dir}/{method}')):
                for file_name in sorted(os.listdir(f'{grasp_dir}/{method}/{contact_mask}')):
                    if file_name.count('.pkl') == 0: continue

                    # Extract info
                    with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name}', 'rb') as file:
                        E_i = pickle.load(file)

                    if i > len(naive_other_estimates) - 1:
                        naive_other_estimates.append(copy.deepcopy(empty_estimate_dict))
                        with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name.split(".")[0]}.json', 'r') as file:
                            config_i = json.load(file)
                        naive_other_configs.append(config_i)
                    
                    naive_other_estimates[i][object_name].append(E_i)
                    i += 1

            # Unpack Hertzian estimations for each config type
            i = 0
            method = 'hertz_other'
            for contact_mask in sorted(os.listdir(f'{grasp_dir}/{method}')):
                for file_name in sorted(os.listdir(f'{grasp_dir}/{method}/{contact_mask}')):
                    if file_name.count('.pkl') == 0: continue

                    # Extract info
                    with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name}', 'rb') as file:
                        E_i = pickle.load(file)

                    if i > len(hertz_other_estimates) - 1:
                        hertz_other_estimates.append(copy.deepcopy(empty_estimate_dict))
                        with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name.split(".")[0]}.json', 'r') as file:
                            config_i = json.load(file)
                        hertz_other_configs.append(config_i)

                    hertz_other_estimates[i][object_name].append(E_i)
                    i += 1

            # Unpack MDR estimations for each config type
            i = 0
            method = 'MDR_other'
            for contact_mask in sorted(os.listdir(f'{grasp_dir}/{method}')):
                for file_name in sorted(os.listdir(f'{grasp_dir}/{method}/{contact_mask}')):
                    if file_name.count('.pkl') == 0: continue

                    # Extract info
                    with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name}', 'rb') as file:
                        E_i = pickle.load(file)

                    if i > len(MDR_other_estimates) - 1:
                        MDR_other_estimates.append(copy.deepcopy(empty_estimate_dict))
                        with open(f'{grasp_dir}/{method}/{contact_mask}/{file_name.split(".")[0]}.json', 'r') as file:
                            config_i = json.load(file)
                        MDR_other_configs.append(config_i)

                    MDR_other_estimates[i][object_name].append(E_i)
                    i += 1

            # Unpack naive estimations using both sensors for each config type
            i = 0
            for contact_mask in sorted(os.listdir(f'{grasp_dir}/naive_both_sides')):
                for file_name in sorted(os.listdir(f'{grasp_dir}/naive_both_sides/{contact_mask}')):
                    if file_name.count('.pkl') == 0: continue

                    # Extract info
                    with open(f'{grasp_dir}/naive_both_sides/{contact_mask}/{file_name}', 'rb') as file:
                        E_i = pickle.load(file)

                    if i > len(naive_both_sides_estimates) - 1:
                        naive_both_sides_estimates.append(copy.deepcopy(empty_estimate_dict))
                        with open(f'{grasp_dir}/naive_both_sides/{contact_mask}/{file_name.split(".")[0]}.json', 'r') as file:
                            config_i = json.load(file)
                        naive_both_sides_configs.append(config_i)
                    
                    naive_both_sides_estimates[i][object_name].append(E_i)
                    i += 1

            # Unpack Hertzian estimations using both sensors for each config type
            i = 0
            for contact_mask in sorted(os.listdir(f'{grasp_dir}/hertz_both_sides')):
                for file_name in sorted(os.listdir(f'{grasp_dir}/hertz_both_sides/{contact_mask}')):
                    if file_name.count('.pkl') == 0: continue

                    # Extract info
                    with open(f'{grasp_dir}/hertz_both_sides/{contact_mask}/{file_name}', 'rb') as file:
                        E_i = pickle.load(file)

                    if i > len(hertz_both_sides_estimates) - 1:
                        hertz_both_sides_estimates.append(copy.deepcopy(empty_estimate_dict))
                        with open(f'{grasp_dir}/hertz_both_sides/{contact_mask}/{file_name.split(".")[0]}.json', 'r') as file:
                            config_i = json.load(file)
                        hertz_both_sides_configs.append(config_i)

                    hertz_both_sides_estimates[i][object_name].append(E_i)
                    i += 1

            # Unpack MDR estimations using both sensors for each config type
            i = 0
            for contact_mask in sorted(os.listdir(f'{grasp_dir}/MDR_both_sides')):
                for file_name in sorted(os.listdir(f'{grasp_dir}/MDR_both_sides/{contact_mask}')):
                    if file_name.count('.pkl') == 0: continue

                    # Extract info
                    with open(f'{grasp_dir}/MDR_both_sides/{contact_mask}/{file_name}', 'rb') as file:
                        E_i = pickle.load(file)

                    if i > len(MDR_both_sides_estimates) - 1:
                        MDR_both_sides_estimates.append(copy.deepcopy(empty_estimate_dict))
                        with open(f'{grasp_dir}/MDR_both_sides/{contact_mask}/{file_name.split(".")[0]}.json', 'r') as file:
                            config_i = json.load(file)
                        MDR_both_sides_configs.append(config_i)

                    MDR_both_sides_estimates[i][object_name].append(E_i)
                    i += 1

    def avg_estimate(estimate, estimate_other):
        if estimate > 0 and estimate_other > 0:
            return (estimate + estimate_other)/2       
        elif estimate > 0:
            return estimate
        else:
            return estimate_other

    naive_avg_estimates = []
    for i in range(len(naive_estimates)):
        naive_avg_estimates.append(copy.deepcopy(empty_estimate_dict))
        for object_name in MDR_estimates[i].keys():
            naive_avg_estimates[-1][object_name] = [ 
                avg_estimate(naive_estimates[i][object_name][j], naive_other_estimates[i][object_name][j]) for j in range(len(naive_estimates[i][object_name]))
            ]
    
    hertz_avg_estimates = []
    for i in range(len(hertz_estimates)):
        hertz_avg_estimates.append(copy.deepcopy(empty_estimate_dict))
        for object_name in MDR_estimates[i].keys():
            hertz_avg_estimates[-1][object_name] = [ 
                avg_estimate(hertz_estimates[i][object_name][j], hertz_other_estimates[i][object_name][j]) for j in range(len(hertz_estimates[i][object_name]))
            ]
    
    MDR_avg_estimates = []
    for i in range(len(MDR_estimates)):
        MDR_avg_estimates.append(copy.deepcopy(empty_estimate_dict))
        for object_name in MDR_estimates[i].keys():
            MDR_avg_estimates[-1][object_name] = [ 
                avg_estimate(MDR_estimates[i][object_name][j], MDR_other_estimates[i][object_name][j]) for j in range(len(MDR_estimates[i][object_name]))
            ]

    # Find a linear scaling for each set of predictions to minimize error
    print('Scaling no tactile...\n')
    no_tactile_estimates = scale_predictions(no_tactile_estimates)
    print('Scaling naive...\n')
    naive_estimates      = [ scale_predictions(x) for x in naive_estimates ]
    print('Scaling Hertzian...\n')
    hertz_estimates      = [ scale_predictions(x) for x in hertz_estimates ]
    print('Scaling MDR...\n')
    MDR_estimates        = [ scale_predictions(x, quadratic=True) for x in MDR_estimates ]
    print('Scaling naive (other)...\n')
    naive_other_estimates      = [ scale_predictions(x) for x in naive_other_estimates ]
    print('Scaling Hertzian (other)...\n')
    hertz_other_estimates      = [ scale_predictions(x) for x in hertz_other_estimates ]
    print('Scaling MDR (other)...\n')
    MDR_other_estimates        = [ scale_predictions(x, quadratic=True) for x in MDR_other_estimates ]
    print('Scaling naive (both sides)...\n')
    naive_both_sides_estimates      = [ scale_predictions(x) for x in naive_both_sides_estimates ]
    print('Scaling Hertzian (both sides)...\n')
    hertz_both_sides_estimates      = [ scale_predictions(x) for x in hertz_both_sides_estimates ]
    print('Scaling MDR (both sides)...\n')
    MDR_both_sides_estimates        = [ scale_predictions(x, quadratic=True) for x in MDR_both_sides_estimates ]
    print('Scaling naive (avg)...\n')
    naive_avg_estimates         = [ scale_predictions(x) for x in naive_avg_estimates ]
    print('Scaling Hertzian (avg)...\n')
    hertz_avg_estimates         = [ scale_predictions(x) for x in hertz_avg_estimates ]
    print('Scaling MDR (avg)...\n')
    MDR_avg_estimates           = [ scale_predictions(x, quadratic=True) for x in MDR_avg_estimates ]

    # Evaluate each set of estimates and pick the best
    no_tactile_stats = compute_estimation_stats(no_tactile_estimates)
    naive_stats = [
        compute_estimation_stats(naive_estimates[i]) for i in range(len(naive_estimates))
    ]
    hertz_stats = [
        compute_estimation_stats(hertz_estimates[i]) for i in range(len(hertz_estimates))
    ]
    MDR_stats = [
        compute_estimation_stats(MDR_estimates[i]) for i in range(len(MDR_estimates))
    ]
    naive_other_stats = [
        compute_estimation_stats(naive_other_estimates[i]) for i in range(len(naive_other_estimates))
    ]
    hertz_other_stats = [
        compute_estimation_stats(hertz_other_estimates[i]) for i in range(len(hertz_other_estimates))
    ]
    MDR_other_stats = [
        compute_estimation_stats(MDR_other_estimates[i]) for i in range(len(MDR_other_estimates))
    ]
    naive_both_sides_stats = [
        compute_estimation_stats(naive_both_sides_estimates[i]) for i in range(len(naive_both_sides_estimates))
    ]
    hertz_both_sides_stats = [
        compute_estimation_stats(hertz_both_sides_estimates[i]) for i in range(len(hertz_both_sides_estimates))
    ]
    MDR_both_sides_stats = [
        compute_estimation_stats(MDR_both_sides_estimates[i]) for i in range(len(MDR_both_sides_estimates))
    ]
    naive_avg_stats = [
        compute_estimation_stats(naive_avg_estimates[i]) for i in range(len(naive_avg_estimates))
    ]
    hertz_avg_stats = [
        compute_estimation_stats(hertz_avg_estimates[i]) for i in range(len(hertz_avg_estimates))
    ]
    MDR_avg_stats = [
        compute_estimation_stats(MDR_avg_estimates[i]) for i in range(len(MDR_avg_estimates))
    ]

    # Sort based on log difference
    naive_i_order               = sorted(range(len(naive_stats)), key=lambda i: naive_stats[i]['avg_log_diff'])
    hertz_i_order               = sorted(range(len(hertz_stats)), key=lambda i: hertz_stats[i]['avg_log_diff'])
    MDR_i_order                 = sorted(range(len(MDR_stats)), key=lambda i: MDR_stats[i]['avg_log_diff'])
    naive_other_i_order         = sorted(range(len(naive_stats)), key=lambda i: naive_stats[i]['avg_log_diff'])
    hertz_other_i_order         = sorted(range(len(hertz_stats)), key=lambda i: hertz_stats[i]['avg_log_diff'])
    MDR_other_i_order           = sorted(range(len(MDR_stats)), key=lambda i: MDR_stats[i]['avg_log_diff'])
    naive_both_sides_i_order    = sorted(range(len(naive_both_sides_stats)), key=lambda i: naive_both_sides_stats[i]['avg_log_diff'])
    hertz_both_sides_i_order    = sorted(range(len(hertz_both_sides_stats)), key=lambda i: hertz_both_sides_stats[i]['avg_log_diff'])
    MDR_both_sides_i_order      = sorted(range(len(MDR_both_sides_stats)), key=lambda i: MDR_both_sides_stats[i]['avg_log_diff'])
    naive_avg_i_order           = sorted(range(len(naive_avg_stats)), key=lambda i: naive_avg_stats[i]['avg_log_diff'])
    hertz_avg_i_order           = sorted(range(len(hertz_avg_stats)), key=lambda i: hertz_avg_stats[i]['avg_log_diff'])
    MDR_avg_i_order             = sorted(range(len(MDR_avg_stats)), key=lambda i: MDR_avg_stats[i]['avg_log_diff'])

    naive_configs_ordered               = [ naive_configs[i] for i in naive_i_order ]
    hertz_configs_ordered               = [ hertz_configs[i] for i in hertz_i_order ]
    MDR_configs_ordered                 = [ MDR_configs[i] for i in MDR_i_order ]
    naive_other_configs_ordered         = [ naive_other_configs[i] for i in naive_other_i_order ]
    hertz_other_configs_ordered         = [ hertz_other_configs[i] for i in hertz_other_i_order ]
    MDR_other_configs_ordered           = [ MDR_other_configs[i] for i in MDR_other_i_order ]
    naive_both_sides_configs_ordered    = [ naive_both_sides_configs[i] for i in naive_both_sides_i_order ]
    hertz_both_sides_configs_ordered    = [ hertz_both_sides_configs[i] for i in hertz_both_sides_i_order ]
    MDR_both_sides_configs_ordered      = [ MDR_both_sides_configs[i] for i in MDR_both_sides_i_order ]
    naive_avg_configs_ordered           = [ naive_configs[i] for i in naive_avg_i_order ]
    hertz_avg_configs_ordered           = [ hertz_configs[i] for i in hertz_avg_i_order ]
    MDR_avg_configs_ordered             = [ MDR_configs[i] for i in MDR_avg_i_order ]

    naive_stats_ordered                 = [ naive_stats[i] for i in naive_i_order ]
    hertz_stats_ordered                 = [ hertz_stats[i] for i in hertz_i_order ]
    MDR_stats_ordered                   = [ MDR_stats[i] for i in MDR_i_order ]
    naive_other_stats_ordered           = [ naive_other_stats[i] for i in naive_other_i_order ]
    hertz_other_stats_ordered           = [ hertz_other_stats[i] for i in hertz_other_i_order ]
    MDR_other_stats_ordered             = [ MDR_other_stats[i] for i in MDR_other_i_order ]
    naive_both_sides_stats_ordered      = [ naive_both_sides_stats[i] for i in naive_both_sides_i_order ]
    hertz_both_sides_stats_ordered      = [ hertz_both_sides_stats[i] for i in hertz_both_sides_i_order ]
    MDR_both_sides_stats_ordered        = [ MDR_both_sides_stats[i] for i in MDR_both_sides_i_order ]
    naive_avg_stats_ordered             = [ naive_avg_stats[i] for i in naive_avg_i_order ]
    hertz_avg_stats_ordered             = [ hertz_avg_stats[i] for i in hertz_avg_i_order ]
    MDR_avg_stats_ordered               = [ MDR_avg_stats[i] for i in MDR_avg_i_order ]

    obj_prediction_log_diff, obj_avg_log_diff = compute_object_performance([ naive_avg_estimates[naive_avg_i_order[0]],  \
                                                                             MDR_avg_estimates[MDR_avg_i_order[0]]     ])

    object_names = list(obj_prediction_log_diff.keys())
    obj_i_ordered = sorted(range(len(object_names)), key=lambda i: obj_avg_log_diff[object_names[i]])
    obj_avg_log_diff_ordered = [ (object_names[i], obj_avg_log_diff[object_names[i]]) for i in obj_i_ordered ]
    obj_predictions_log_diff_ordered = [ (object_names[i], obj_prediction_log_diff[object_names[i]]) for i in obj_i_ordered ]

    contact_mask_stats = compute_contact_mask_performance(naive_configs + MDR_configs, naive_stats + MDR_stats)

    naive_shape_log_diff, naive_shape_log_acc       = compute_shape_performance(naive_avg_estimates[naive_avg_i_order[0]])
    naive_material_log_diff, naive_material_log_acc = compute_material_performance(naive_avg_estimates[naive_avg_i_order[0]])
    hertz_shape_log_diff, hertz_shape_log_acc       = compute_shape_performance(hertz_avg_estimates[hertz_avg_i_order[0]])
    hertz_material_log_diff, hertz_material_log_acc = compute_material_performance(hertz_avg_estimates[hertz_avg_i_order[0]])
    MDR_shape_log_diff, MDR_shape_log_acc           = compute_shape_performance(MDR_avg_estimates[MDR_avg_i_order[0]])
    MDR_material_log_diff, MDR_material_log_acc     = compute_material_performance(MDR_avg_estimates[MDR_avg_i_order[0]])

    # Create plots showing how well each method does
    print('Plotting no tactile...')
    plot_performance('No Tactile', no_tactile_estimates, object_to_modulus)
    print('Plotting naive...')
    plot_performance('Naive Elasticity Method', naive_estimates[naive_i_order[0]], object_to_modulus)
    print('Plotting Hertzian...')
    plot_performance('Hertzian Method', hertz_estimates[hertz_i_order[0]], object_to_modulus)
    print('Plotting MDR...')
    plot_performance('MDR', MDR_estimates[MDR_i_order[0]], object_to_modulus)
    print('Plotting naive (other)...')
    plot_performance('Naive Elasticity (other side)', naive_other_estimates[naive_other_i_order[0]], object_to_modulus)
    print('Plotting Hertzian (other)...')
    plot_performance('Hertzian Method (other side)', hertz_other_estimates[hertz_other_i_order[0]], object_to_modulus)
    print('Plotting MDR (other)...')
    plot_performance('MDR (other side)', MDR_other_estimates[MDR_other_i_order[0]], object_to_modulus)
    print('Plotting naive (both sides)...')
    plot_performance('Naive Elasticity (Both Sides)', naive_both_sides_estimates[naive_both_sides_i_order[0]], object_to_modulus)
    print('Plotting Hertzian (both sides)...')
    plot_performance('Hertzian (Both Sides)', hertz_both_sides_estimates[hertz_both_sides_i_order[0]], object_to_modulus)
    print('Plotting MDR (both sides)...')
    plot_performance('MDR (Both Sides)', MDR_both_sides_estimates[MDR_both_sides_i_order[0]], object_to_modulus)
    print('Plotting naive (avg)...')
    plot_performance('Naive Elasticity (avg of sides)', naive_avg_estimates[naive_avg_i_order[0]], object_to_modulus)
    print('Plotting Hertzian (avg)...')
    plot_performance('Hertzian (avg of sides)', hertz_avg_estimates[hertz_avg_i_order[0]], object_to_modulus)
    print('Plotting MDR (avg)...')
    plot_performance('MDR (avg of sides)', MDR_avg_estimates[MDR_avg_i_order[0]], object_to_modulus)
    print('Done.')
    
    # materials = ['Foam', 'Rubber', 'Plastic', 'Rubber', 'Wood', 'Metal', 'Ceramic', 'Glass', 'Paper', 'Food']
    
    # no_tactile_material_predictions = { x:[] for x in materials }
    # naive_material_predictions = { x:[] for x in materials }
    # hertz_material_predictions = { x:[] for x in materials }
    # MDR_material_predictions = { x:[] for x in materials }
    # naive_material_labels = { x:[] for x in materials }
    # hertz_material_labels = { x:[] for x in materials }
    # MDR_material_labels = { x:[] for x in materials }

    # for object_name in no_tactile_estimates.keys():   
    #     mat = object_to_material[object_name]
    #     no_tactile_material_predictions[mat].extend(no_tactile_estimates[object_name])
    
    # class NumpyEncoder(json.JSONEncoder):
    #     def default(self, obj):
    #         if isinstance(obj, np.float32):
    #             return float(obj)
    #         return json.JSONEncoder.default(self, obj)

    # with open(f'./plotting_data/no_tactile_predictions.json', 'w') as json_file:
    #     json.dump(no_tactile_material_predictions, json_file, cls=NumpyEncoder)

    # for object_name in tqdm(os.listdir('./data/training_estimations_nan_filtered')):   
    #     for folder_name in os.listdir(f'./data/training_estimations_nan_filtered/{object_name}'):
    #         try:
    #             with open(f'./data/training_estimations_nan_filtered/{object_name}/{folder_name}/E.pkl', 'rb') as file:
    #                 E = pickle.load(file)

    #             if not (object_to_material[object_name] in materials): continue
                
    #             mat = object_to_material[object_name]
                    
    #             naive_material_predictions[mat].append(E[0])
    #             hertz_material_predictions[mat].append(E[1])
    #             MDR_material_predictions[mat].append(E[2])
    #             naive_material_labels[mat].append(object_to_modulus[object_name])
    #             hertz_material_labels[mat].append(object_to_modulus[object_name])
    #             MDR_material_labels[mat].append(object_to_modulus[object_name])

    #         except:
    #             print('skip')
    
    # class NumpyEncoder(json.JSONEncoder):
    #     def default(self, obj):
    #         if isinstance(obj, np.float32):
    #             return float(obj)
    #         return json.JSONEncoder.default(self, obj)

    # with open(f'./plotting_data/naive_predictions.json', 'w') as json_file:
    #     json.dump(naive_material_predictions, json_file, cls=NumpyEncoder)
    # with open(f'./plotting_data/naive_labels.json', 'w') as json_file:
    #     json.dump(naive_material_labels, json_file, cls=NumpyEncoder)
    # with open(f'./plotting_data/hertz_predictions.json', 'w') as json_file:
    #     json.dump(hertz_material_predictions, json_file, cls=NumpyEncoder)
    # with open(f'./plotting_data/hertz_labels.json', 'w') as json_file:
    #     json.dump(hertz_material_labels, json_file, cls=NumpyEncoder)
    # with open(f'./plotting_data/MDR_predictions.json', 'w') as json_file:
    #     json.dump(MDR_material_predictions, json_file, cls=NumpyEncoder)
    # with open(f'./plotting_data/MDR_labels.json', 'w') as json_file:
    #     json.dump(MDR_material_labels, json_file, cls=NumpyEncoder)

    # # Write training estimations
    # skipped = 0; total = 0
    # for object_name in naive_avg_estimates[naive_avg_i_order[0]].keys():
    #     if object_name in EXCLUDE: continue
    #     if not os.path.exists(f'{DATA_DIR}/training_estimations_nan_filtered/{object_name}'):
    #         os.mkdir(f'{DATA_DIR}/training_estimations_nan_filtered/{object_name}')

    #     for t in range(len(naive_avg_estimates[naive_avg_i_order[0]][object_name])):
            
    #         total += 1
    #         if (not naive_avg_estimates[naive_avg_i_order[0]][object_name][t] > 0) or \
    #             (not naive_avg_estimates[naive_avg_i_order[0]][object_name][t] > 0) or \
    #             (not naive_avg_estimates[naive_avg_i_order[0]][object_name][t] > 0):
    #             skipped += 1
    #             continue

    #         E_naive = closest_non_nan_element(naive_avg_estimates[naive_avg_i_order[0]][object_name], t)
    #         E_hertz = closest_non_nan_element(hertz_avg_estimates[hertz_avg_i_order[0]][object_name], t)
    #         E_MDR = closest_non_nan_element(MDR_avg_estimates[MDR_avg_i_order[0]][object_name], t)
    #         assert E_naive > 0 and E_hertz > 0 and E_MDR > 0

    #         if not os.path.exists(f'{DATA_DIR}/training_estimations_nan_filtered/{object_name}/t={t}'):
    #             os.mkdir(f'{DATA_DIR}/training_estimations_nan_filtered/{object_name}/t={t}')

    #         E_estimates = np.array([E_naive, E_hertz, E_MDR])
    #         with open(f'{DATA_DIR}/training_estimations_nan_filtered/{object_name}/t={t}/E.pkl', 'wb') as file:
    #             pickle.dump(E_estimates, file)

    # print('done')