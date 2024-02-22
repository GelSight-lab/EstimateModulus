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
            'red_foam_brick', 'blue_foam_brick', 'green_foam_brick', # 'yellow_foam_brick',
            'apple', 'orange', 'strawberry', 'ripe_banana', 'unripe_banana', 
            'lacrosse_ball', 'scotch_brite', 'fake_washer_stack', 'cork',
            'baseball', 'plastic_measuring_cup', 'whiteboard_eraser', 'lifesaver_hard', 'cutting_board',
            'plastic_knife', 'plastic_fork', 'plastic_spoon', 'plastic_fork_white',
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
def scale_predictions(prediction_dict, scale_cutoff=1e11):
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

    poly = np.polyfit(np.log10(x_filtered), np.log10(y_filtered), 1)

    # Scale all predictions accordingly
    scaled_prediction_dict = {}
    for object_name in prediction_dict.keys():
        scaled_prediction_dict[object_name] = []
        if object_name in EXCLUDE: continue
        for E in prediction_dict[object_name]:
            if E > 0 and not math.isnan(E):
                E_scaled = 10**(np.log10(E)*poly[0] + poly[1])
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
        'log_accuracy': np.sum(log_diff < 0.5) / len(log_diff),
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
    plt.fill_between([100, 10**12], [10**(1.5), 10**(11.5)], [10**(2.5), 10**(12.5)], color='gray', alpha=0.2)
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
    naive_estimates         = []
    naive_configs           = []
    hertz_estimates         = []
    hertz_configs           = []
    MDR_estimates           = []
    MDR_configs             = []
    stochastic_estimates    = []
    stochastic_configs      = []

    naive_both_sides_estimates     = []
    naive_both_sides_configs       = []
    hertz_both_sides_estimates     = []
    hertz_both_sides_configs       = []
    MDR_both_sides_estimates       = []
    MDR_both_sides_configs         = []

    for object_name in tqdm(sorted(os.listdir(f'{DATA_DIR}/estimations'))):
        if object_name.count('.') > 0: continue
        if object_name in EXCLUDE: continue

        for trial_folder in os.listdir(f'{DATA_DIR}/estimations/{object_name}'):
            grasp_dir = f'{DATA_DIR}/estimations/{object_name}/{trial_folder}'

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
                    with open(f'{grasp_dir}/MDR/{contact_mask}/{file_name}', 'rb') as file:
                        E_i = pickle.load(file)

                    if i > len(MDR_estimates) - 1:
                        MDR_estimates.append(copy.deepcopy(empty_estimate_dict))
                        with open(f'{grasp_dir}/MDR/{contact_mask}/{file_name.split(".")[0]}.json', 'r') as file:
                            config_i = json.load(file)
                        MDR_configs.append(config_i)

                    MDR_estimates[i][object_name].append(E_i)
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

            # # Unpack stochastic estimation
            # with open(f'{grasp_dir}/stochastic/E.pkl', 'rb') as file:
            #     E_i = pickle.load(file)
            # stochastic_estimates[0][object_name].append(E_i)


    # Find a linear scaling for each set of predictions to minimize error
    print('Scaling naive...\n')
    naive_estimates      = [ scale_predictions(x) for x in naive_estimates ]
    print('Scaling Hertzian...\n')
    hertz_estimates      = [ scale_predictions(x) for x in hertz_estimates ]
    print('Scaling MDR...\n')
    MDR_estimates        = [ scale_predictions(x) for x in MDR_estimates ]
    print('Scaling naive (both sides)...\n')
    naive_both_sides_estimates      = [ scale_predictions(x) for x in naive_both_sides_estimates ]
    print('Scaling Hertzian (both sides)...\n')
    hertz_both_sides_estimates      = [ scale_predictions(x) for x in hertz_both_sides_estimates ]
    print('Scaling MDR (both sides)...\n')
    MDR_both_sides_estimates        = [ scale_predictions(x) for x in MDR_both_sides_estimates ]
    # print('Scaling stochastic...\n')
    # stochastic_estimates = [ scale_predictions(x, object_to_modulus) for x in stochastic_estimates ]

    # Evaluate each set of estimates and pick the best
    naive_stats = [
        compute_estimation_stats(naive_estimates[i]) for i in range(len(naive_estimates))
    ]
    hertz_stats = [
        compute_estimation_stats(hertz_estimates[i]) for i in range(len(hertz_estimates))
    ]
    MDR_stats = [
        compute_estimation_stats(MDR_estimates[i]) for i in range(len(MDR_estimates))
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
    # stochastic_stats = [
    #     compute_estimation_stats(stochastic_estimates[i], object_to_modulus) for i in range(len(stochastic_estimates))
    # ]

    # Sort based on log difference
    naive_i_order       = sorted(range(len(naive_stats)), key=lambda i: naive_stats[i]['avg_log_diff'])
    hertz_i_order       = sorted(range(len(hertz_stats)), key=lambda i: hertz_stats[i]['avg_log_diff'])
    MDR_i_order         = sorted(range(len(MDR_stats)), key=lambda i: MDR_stats[i]['avg_log_diff'])
    naive_both_sides_i_order       = sorted(range(len(naive_both_sides_stats)), key=lambda i: naive_both_sides_stats[i]['avg_log_diff'])
    hertz_both_sides_i_order       = sorted(range(len(hertz_both_sides_stats)), key=lambda i: hertz_both_sides_stats[i]['avg_log_diff'])
    MDR_both_sides_i_order         = sorted(range(len(MDR_both_sides_stats)), key=lambda i: MDR_both_sides_stats[i]['avg_log_diff'])
    # stochastic_i_order  = sorted(range(len(stochastic_stats)), key=lambda i: stochastic_stats[i]['avg_log_diff'])

    naive_configs_ordered       = [ naive_configs[i] for i in naive_i_order ]
    hertz_configs_ordered       = [ hertz_configs[i] for i in hertz_i_order ]
    MDR_configs_ordered         = [ MDR_configs[i] for i in MDR_i_order ]
    naive_both_sides_configs_ordered       = [ naive_both_sides_configs[i] for i in naive_both_sides_i_order ]
    hertz_both_sides_configs_ordered       = [ hertz_both_sides_configs[i] for i in hertz_both_sides_i_order ]
    MDR_both_sides_configs_ordered         = [ MDR_both_sides_configs[i] for i in MDR_both_sides_i_order ]

    naive_stats_ordered         = [ naive_stats[i] for i in naive_i_order ]
    hertz_stats_ordered         = [ hertz_stats[i] for i in hertz_i_order ]
    MDR_stats_ordered           = [ MDR_stats[i] for i in MDR_i_order ]
    naive_both_sides_stats_ordered         = [ naive_both_sides_stats[i] for i in naive_both_sides_i_order ]
    hertz_both_sides_stats_ordered         = [ hertz_both_sides_stats[i] for i in hertz_both_sides_i_order ]
    MDR_both_sides_stats_ordered           = [ MDR_both_sides_stats[i] for i in MDR_both_sides_i_order ]

    obj_prediction_log_diff, obj_avg_log_diff = compute_object_performance([ naive_both_sides_estimates[naive_i_order[0]],  \
                                                                             MDR_both_sides_estimates[MDR_i_order[0]]     ])

    object_names = list(obj_prediction_log_diff.keys())
    obj_i_ordered = sorted(range(len(object_names)), key=lambda i: obj_avg_log_diff[object_names[i]])
    obj_avg_log_diff_ordered = [ (object_names[i], obj_avg_log_diff[object_names[i]]) for i in obj_i_ordered ]
    obj_predictions_log_diff_ordered = [ (object_names[i], obj_prediction_log_diff[object_names[i]]) for i in obj_i_ordered ]

    contact_mask_stats = compute_contact_mask_performance(naive_configs + MDR_configs, naive_stats + MDR_stats)
    
    # Create plots showing how well each method does
    print('Plotting naive...')
    plot_performance('Naive Elasticity Method', naive_estimates[naive_i_order[0]], object_to_modulus)
    print('Plotting Hertzian...')
    plot_performance('Hertzian Method', hertz_estimates[hertz_i_order[0]], object_to_modulus)
    print('Plotting MDR...')
    plot_performance('MDR', MDR_estimates[MDR_i_order[0]], object_to_modulus)
    print('Plotting naive (both sides)...')
    plot_performance('Naive Elasticity (Both Sides)', naive_both_sides_estimates[naive_both_sides_i_order[0]], object_to_modulus)
    print('Plotting Hertzian (both sides)...')
    plot_performance('Hertzian (Both Sides)', hertz_both_sides_estimates[hertz_both_sides_i_order[0]], object_to_modulus)
    print('Plotting MDR (both sides)...')
    plot_performance('MDR (Both Sides)', MDR_both_sides_estimates[MDR_both_sides_i_order[0]], object_to_modulus)
    # print('Plotting stochastic...')
    # plot_performance(r'Stochastic Method', stochastic_estimates[0], object_to_modulus)
    print('Done.')

    # # Write training estimations
    # for object_name in naive_estimates[naive_i_order[0]].keys():
    #     if object_name in EXCLUDE: continue
    #     if not os.path.exists(f'{DATA_DIR}/training_estimations/{object_name}'):
    #         os.mkdir(f'{DATA_DIR}/training_estimations/{object_name}')

    #     for t in range(len(naive_estimates[naive_i_order[0]][object_name])):
            
    #         E_naive = closest_non_nan_element(naive_estimates[naive_i_order[0]][object_name], t)
    #         E_hertz = closest_non_nan_element(hertz_estimates[hertz_i_order[0]][object_name], t)
    #         E_MDR = closest_non_nan_element(MDR_estimates[MDR_i_order[0]][object_name], t)
    #         assert E_naive > 0 and E_hertz > 0 and E_MDR > 0

    #         if not os.path.exists(f'{DATA_DIR}/training_estimations/{object_name}/t={t}'):
    #             os.mkdir(f'{DATA_DIR}/training_estimations/{object_name}/t={t}')

    #         E_estimates = np.array([E_naive, E_hertz, E_MDR])
    #         with open(f'{DATA_DIR}/training_estimations/{object_name}/t={t}/E.pkl', 'wb') as file:
    #             pickle.dump(E_estimates, file)