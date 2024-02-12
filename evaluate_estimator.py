import os
import csv
import json
import pickle
import copy
from tactile_estimate import *

def random_hex_color():
    # Generate random values for red, green, and blue
    R = random.randint(0, 255)
    G = random.randint(0, 255)
    B = random.randint(0, 255)

    # Format the values as hexadecimal and concatenate them
    return "#{:02X}{:02X}{:02X}".format(R, G, B)

wedge_video         = GelsightWedgeVideo(config_csv="./config_100.csv") # Force-sensing finger
other_wedge_video   = GelsightWedgeVideo(config_csv="./config_200_markers.csv") # Non-sensing finger
contact_force       = ContactForce()
gripper_width       = GripperWidth()
grasp_data          = GraspData(wedge_video=wedge_video, other_wedge_video=other_wedge_video, contact_force=contact_force, gripper_width=gripper_width, use_gripper_width=True)

DATA_DIR = './data'
RUN_NAME = 'THRESHOLD'

# Objects to exclude from evaluation
EXCLUDE = ['playdoh', 'silly_putty', 'silly_puty', 'racquet_ball']

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
def scale_predictions(prediction_dict, label_dict, linear_log_fit=True, exp_fit=False):
    assert linear_log_fit ^ exp_fit
    x, y = [], []
    for object_name in prediction_dict.keys():
        if object_name in EXCLUDE: continue
        for E in prediction_dict[object_name]:
            if E > 0 and not math.isnan(E):
                x.append(E)
                y.append(label_dict[object_name])

    # Filter out outliers for fitting
    x, y = np.array(x), np.array(y)
    outlier_mask = np.abs(x - np.mean(x)) < 2*np.std(x)
    x_filtered = x[outlier_mask]
    y_filtered = y[outlier_mask]

    # TODO: Make a better fit here
    if linear_log_fit:
        poly = np.polyfit(np.log10(x_filtered), np.log10(y_filtered), 1)
    elif exp_fit:
        poly = np.polyfit(x_filtered, np.log10(y_filtered), 1)
    
    # Scale all predictions accordingly
    scaled_prediction_dict = {}
    for object_name in prediction_dict.keys():
        scaled_prediction_dict[object_name] = []
        if object_name in EXCLUDE: continue
        for E in prediction_dict[object_name]:
            if E > 0 and not math.isnan(E):
                if linear_log_fit:
                    E_scaled = 10**(np.log10(E)*poly[0] + poly[1])
                elif exp_fit:
                    E_scaled = poly[0]*np.exp(poly[0]*E)
                scaled_prediction_dict[object_name].append(E_scaled)
            else:
                scaled_prediction_dict[object_name].append(E)

    return scaled_prediction_dict

# Compute statistics to evaluate the performance of estimation method
def compute_estimation_stats(prediction_dict, label_dict):
    log_diff = []
    for object_name in prediction_dict.keys():
        if object_name in EXCLUDE: continue
        for E in prediction_dict[object_name]:
            if E > 0 and not math.isnan(E):
                assert not math.isnan(E)
                log_diff.append(abs(np.log10(E) - np.log10(label_dict[object_name])))
    
    log_diff = np.array(log_diff)
    log_diff = log_diff[~np.isnan(log_diff)]
    return {
        'avg_log_diff': log_diff.mean(),
        'log_accuracy': np.sum(log_diff < 0.5) / len(log_diff),
    }

# Plot on log scale to see how performance is
def plot_performance(plot_title, prediction_dict, label_dict):
    material_to_color = {
        'Foam': 'firebrick',
        'Plastic': 'forestgreen',
        'Wood': 'goldenrod',
        'Paper': 'darkviolet',
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
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rc('text', usetex=True)
    plt.figure()
    plt.plot([100, 10**12], [100, 10**12], 'k--', label='_')
    plt.fill_between([100, 10**12], [10**(1.5), 10**(11.5)], [10**(2.5), 10**(12.5)], color='gray', alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')

    for mat in material_to_color.keys():
        plt.plot(material_label_data[mat], material_prediction_data[mat], '.', markersize=10, color=material_to_color[mat], label=mat)

    plt.xlabel("Ground Truth Modulus ($E$) [$\\frac{N}{m^2}$]", fontsize=12)
    plt.ylabel("Predicted Modulus ($\\widetilde{E}$) [$\\frac{N}{m^2}$]", fontsize=12)
    plt.xlim([100, 10**12])
    plt.ylim([100, 10**12])
    plt.title(plot_title, fontsize=14)

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    plt.tick_params(axis='both', which='both', labelsize=10)

    plt.savefig(f'{plot_title}.png')
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
    stochastic_estimates    = [empty_estimate_dict.copy()]

    for object_name in sorted(os.listdir(f'{DATA_DIR}/estimations')):
        if object_name.count('.') > 0: continue
        if object_name in EXCLUDE: continue

        for trial_folder in os.listdir(f'{DATA_DIR}/estimations/{object_name}'):

            # Unpack naive estimations for each config type
            for file_name in sorted(os.listdir(f'{DATA_DIR}/estimations/{object_name}/{trial_folder}/naive')):
                if file_name.count('.pkl') == 0: continue

                # Extract info
                i = int(file_name.split('.')[0])
                with open(f'{DATA_DIR}/estimations/{object_name}/{trial_folder}/naive/{file_name}', 'rb') as file:
                    E_i = pickle.load(file)

                if i > len(naive_estimates) - 1:
                    new_dict = copy.deepcopy(empty_estimate_dict)
                    naive_estimates.append(new_dict)
                    with open(f'{DATA_DIR}/estimations/{object_name}/{trial_folder}/naive/{i}.json', 'r') as file:
                        config_i = json.load(file)
                    naive_configs.append(config_i)
                
                naive_estimates[i][object_name].append(E_i)

            # Unpack Hertzian estimations for each config type
            for file_name in sorted(os.listdir(f'{DATA_DIR}/estimations/{object_name}/{trial_folder}/hertz')):
                if file_name.count('.pkl') == 0: continue

                # Extract info
                i = int(file_name.split('.')[0])
                with open(f'{DATA_DIR}/estimations/{object_name}/{trial_folder}/hertz/{file_name}', 'rb') as file:
                    E_i = pickle.load(file)

                if i > len(hertz_estimates) - 1:
                    hertz_estimates.append(empty_estimate_dict.copy())
                    with open(f'{DATA_DIR}/estimations/{object_name}/{trial_folder}/hertz/{i}.json', 'r') as file:
                        config_i = json.load(file)
                    hertz_configs.append(config_i)

                hertz_estimates[i][object_name].append(E_i)

            # Unpack MDR estimations for each config type
            for file_name in sorted(os.listdir(f'{DATA_DIR}/estimations/{object_name}/{trial_folder}/MDR')):
                if file_name.count('.pkl') == 0: continue

                # Extract info
                i = int(file_name.split('.')[0])
                with open(f'{DATA_DIR}/estimations/{object_name}/{trial_folder}/MDR/{file_name}', 'rb') as file:
                    E_i = pickle.load(file)

                if i > len(MDR_estimates) - 1:
                    MDR_estimates.append(empty_estimate_dict.copy())
                    with open(f'{DATA_DIR}/estimations/{object_name}/{trial_folder}/MDR/{i}.json', 'r') as file:
                        config_i = json.load(file)
                    MDR_configs.append(config_i)

                MDR_estimates[i][object_name].append(E_i)

            # Unpack stochastic estimation
            with open(f'{DATA_DIR}/estimations/{object_name}/{trial_folder}/stochastic/E.pkl', 'rb') as file:
                E_i = pickle.load(file)
            stochastic_estimates[0][object_name].append(E_i)

    # Find a linear scaling for each set of predictions to minimize error
    naive_estimates      = [ scale_predictions(x, object_to_modulus) for x in naive_estimates ]
    hertz_estimates      = [ scale_predictions(x, object_to_modulus) for x in hertz_estimates ]
    MDR_estimates        = [ scale_predictions(x, object_to_modulus) for x in MDR_estimates ]
    stochastic_estimates = [ scale_predictions(stochastic_estimates[0], object_to_modulus) ]

    # Evaluate each set of estimates and pick the best
    naive_stats = [
        compute_estimation_stats(naive_estimates[i], object_to_modulus) for i in range(len(naive_estimates))
    ]
    hertz_stats = [
        compute_estimation_stats(hertz_estimates[i], object_to_modulus) for i in range(len(hertz_estimates))
    ]
    MDR_stats = [
        compute_estimation_stats(MDR_estimates[i], object_to_modulus) for i in range(len(MDR_estimates))
    ]
    stochastic_stats = [
        compute_estimation_stats(stochastic_estimates[0], object_to_modulus)
    ]

    # Sort based on log difference
    naive_i_order  = sorted(range(len(naive_stats)), key=lambda i: naive_stats[i]['avg_log_diff'])
    hertz_i_order  = sorted(range(len(hertz_stats)), key=lambda i: hertz_stats[i]['avg_log_diff'])
    MDR_i_order    = sorted(range(len(MDR_stats)), key=lambda i: MDR_stats[i]['avg_log_diff'])

    # Create plots showing how well each method does
    plot_performance('Naive Elasticity Method', naive_estimates[naive_i_order[0]], object_to_modulus)
    plot_performance('Hertzian Method', hertz_estimates[hertz_i_order[0]], object_to_modulus)
    plot_performance('MDR', MDR_estimates[MDR_i_order[0]], object_to_modulus)
    plot_performance('Stochastic Method', stochastic_estimates[0], object_to_modulus)