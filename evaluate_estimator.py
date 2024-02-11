import os
import csv
import json
import pickle
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
def scale_predictions(prediction_dict, label_dict):
    x, y = [], []
    for object_name in prediction_dict.keys():
        if object_name in EXCLUDE: continue
        for E in prediction_dict[object_name]:
            if E > 0:
                x.append(E)
                y.append(label_dict[object_name])
    return np.polyfit(x, y, 1)

# Compute statistics to evaluate the performance of estimation method
def compute_estimation_stats(prediction_dict, linear_scaling, label_dict):
    assert len(linear_scaling) == 2
    loss = []
    log_diff = []
    for object_name in prediction_dict.keys():
        if object_name in EXCLUDE: continue
        for E in prediction_dict[object_name]:
            E_scaled = linear_scaling[0]*E + linear_scaling[1]
            loss.append(abs(E_scaled - label_dict[object_name]))
            log_diff.append(abs(np.log10(E_scaled) - np.log10(label_dict[object_name])))
    
    loss = np.array(loss)
    log_diff = np.array(log_diff)
    return {
        'avg_loss': loss.mean(),
        'avg_log_diff': log_diff.mean(),
        'log_accuracy': np.sum(log_diff < 0.5),
    }

# Plot on log scale to see how performance is
def plot_performance(plot_title, prediction_dict, linear_scaling, label_dict):
    material_to_color = {
        'Foam': 'firebrick',
        'Plastic': 'forestgreen',
        'Wood': 'goldenrod',
        'Glass': 'darkgray',
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
        assert obj in object_to_material.keys()
        mat = object_to_material[obj]
        material_prediction_data[mat].append(linear_scaling[0]*prediction_dict[obj] + linear_scaling[1])
        material_label_data[mat].append(label_dict[obj])

    # Create plot
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure()
    plt.plot([100, 10**12], [100, 10**12], 'k--', label='_')
    plt.fill_between([100, 10**12], [10**(1.5), 10**(11.5)], [10**(2.5), 10**(12.5)], alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')

    for mat in material_to_color.keys():
        plt.plot(material_label_data[mat], material_prediction_data[mat], '.', markersize=10, color=material_to_color[mat], label=mat)

    plt.xlabel("Ground Truth Modulus ($E$) [$\\frac{N}{m^2}$]", fontsize=12)
    plt.ylabel("Predicted Modulus ($\\widetilde{E}$) [$\\frac{N}{m^2}$]", fontsize=12)
    plt.title(plot_title, fontsize=14)

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
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

    for object_name in os.listdir(f'{DATA_DIR}/estimations'):
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
                    naive_estimates.append(empty_estimate_dict.copy())
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
    naive_scalings      = [ scale_predictions(x, object_to_modulus) for x in naive_estimates ]
    hertz_scalings      = [ scale_predictions(x, object_to_modulus) for x in hertz_estimates ]
    MDR_scalings        = [ scale_predictions(x, object_to_modulus) for x in MDR_estimates ]
    stochastic_scalings = [ scale_predictions(stochastic_estimates[0], object_to_modulus) ]

    # Evaluate each set of estimates and pick the best
    naive_stats = [
        compute_estimation_stats(naive_estimates[i], naive_scalings[i], object_to_modulus) for i in range(len(naive_estimates))
    ]
    hertz_stats = [
        compute_estimation_stats(hertz_estimates[i], hertz_scalings[i], object_to_modulus) for i in range(len(hertz_estimates))
    ]
    MDR_stats = [
        compute_estimation_stats(MDR_estimates[i], MDR_scalings[i], object_to_modulus) for i in range(len(MDR_estimates))
    ]
    stochastic_stats = [
        compute_estimation_stats(stochastic_estimates[0], stochastic_scalings[0], object_to_modulus)
    ]

    # Sort based on log difference
    naive_i_order  = sorted(range(len(naive_stats)), key=lambda i: naive_stats[i]['avg_log_diff'])
    hertz_i_order  = sorted(range(len(hertz_stats)), key=lambda i: hertz_stats[i]['avg_log_diff'])
    MDR_i_order    = sorted(range(len(MDR_stats)), key=lambda i: MDR_stats[i]['avg_log_diff'])

    # Create plots showing how well each method does
    compute_estimation_stats('Naive Elasticity Method', naive_estimates[naive_i_order[0]], naive_scalings[naive_i_order[0]], object_to_modulus)
    compute_estimation_stats('Hertzian Method', hertz_estimates[hertz_i_order[0]], hertz_scalings[hertz_i_order[0]], object_to_modulus)
    compute_estimation_stats('MDR', MDR_estimates[MDR_i_order[0]], MDR_scalings[MDR_i_order[0]], object_to_modulus)
    compute_estimation_stats('Stochastic Method', stochastic_estimates[0], stochastic_scalings[0], object_to_modulus)