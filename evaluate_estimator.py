import os
import csv
import json
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

USE_MARKER_FINGER = False
PLOT_DATA = False

# Objects to exclude from evaluation
EXCLUDE = ['playdoh', 'silly_putty', 'silly_puty', 'racquetball']

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

    if PLOT_DATA:
        # Set up raw data plot
        fig1 = plt.figure(1)
        sp1 = fig1.add_subplot(211)
        sp1.set_xlabel('Measured Sensor Deformation (d) [m]')
        sp1.set_ylabel('Force [N]')
        
        # Set up stress / strain axes for naive method
        fig2 = plt.figure(2)
        sp2 = fig2.add_subplot(211)
        sp2.set_xlabel('Strain (dL/L) [/]')
        sp2.set_ylabel('Stress (F/A) [Pa]')

    # Define settings configuration for each estimator method
    naive_config = {
        'contact_mask': None,
        'use_mean': False,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': False,
        'use_lower_resolution_depth': True,
    }
    hertz_config = {
        'contact_mask': None,
        'use_ellipse_mask': False, 
        'fit_mask_to_ellipse': False,
        'use_lower_resolution_depth': True,
    }
    MDR_config = {
        'contact_mask': None,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': False,
        'use_apparent_deformation': True,
        'use_lower_resolution_depth': True,
        'use_mean_radius': False
    }
    stochastic_config = {}

    naive_estimates         = {}
    hertz_estimates         = {}
    MDR_estimates           = {}
    stochastic_estimates    = {}

    max_depths = {}
    skipped_files = []

    estimator = EstimateModulus(grasp_data=grasp_data, use_gripper_width=True, use_other_video=USE_MARKER_FINGER)

    objects = sorted(os.listdir(f'{DATA_DIR}/raw_data'))
    for object_name in objects:
        if PLOT_DATA: plotting_color = random_hex_color()

        # Create list of estimates for each trial of each object
        naive_estimates[object_name]        = []
        hertz_estimates[object_name]        = []
        MDR_estimates[object_name]          = []
        stochastic_estimates[object_name]   = []

        data_files = sorted(os.listdir(f'{DATA_DIR}/raw_data/{object_name}'))
        for file_name in data_files:
            if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_other") > 0:
                continue

            print(f'Working on {os.path.splitext(file_name)[0]}...')

            # Load data into estimator
            estimator._reset_data()
            estimator.load_from_file(f"{DATA_DIR}/raw_data/{object_name}/{os.path.splitext(file_name)[0]}", auto_clip=False)

            # Clip to loading sequence
            estimator.clip_to_press()
            assert len(estimator.depth_images()) == len(estimator.forces()) == len(estimator.gripper_widths())

            # # Save maximum depths
            # max_depths[os.path.splitext(file_name)[0]] = (np.argmax(estimator.max_depths()), estimator.max_depths().max(), -1, estimator.max_depths()[-1])

            # # Skip those with shallow depths
            # if estimator.max_depths().max() <= 10*estimator.depth_threshold:
            #     skipped_files.append(os.path.splitext(file_name)[0])
            #     print('Skipped.')
            #     print('\n')
            #     continue

            # Remove stagnant gripper values across measurement frames
            estimator.interpolate_gripper_widths()

            # Fit using naive estimator
            E_naive = estimator.fit_modulus_naive(
                                    contact_mask=naive_config['contact_mask'],
                                    use_mean=naive_config['use_mean'],
                                    use_ellipse_mask=naive_config['use_ellipse_mask'],
                                    fit_mask_to_ellipse=naive_config['fit_mask_to_ellipse'],
                                    use_lower_resolution_depth=naive_config['use_lower_resolution_depth']
                                )
            naive_estimates[object_name].append(E_naive)
            x_naive = estimator._x_data
            y_naive = estimator._y_data

            if not (E_naive > 0):
                print('Negative or NaN modulus!')

            # Fit using Hertzian estimator
            E_hertz = estimator.fit_modulus_hertz(
                                    contact_mask=hertz_config['contact_mask'],
                                    use_ellipse_mask=hertz_config['use_ellipse_mask'],
                                    fit_mask_to_ellipse=hertz_config['fit_mask_to_ellipse'],
                                    use_lower_resolution_depth=hertz_config['use_lower_resolution_depth']
                                )
            hertz_estimates[object_name].append(E_hertz)

            # Fit using MDR estimator
            E_MDR = estimator.fit_modulus_MDR(
                                    contact_mask=MDR_config['contact_mask'],
                                    use_ellipse_mask=MDR_config['use_ellipse_mask'],
                                    fit_mask_to_ellipse=MDR_config['fit_mask_to_ellipse'],
                                    use_apparent_deformation=MDR_config['use_apparent_deformation'],
                                    use_lower_resolution_depth=MDR_config['use_lower_resolution_depth'],
                                    use_mean_radius=MDR_config['use_mean_radius'],
                                )
            MDR_estimates[object_name].append(E_MDR)

            if E_MDR < 0:
                print('Negative modulus!')

            # Fit using the stochastic estimator
            E_stochastic = estimator.fit_modulus_stochastic()
            stochastic_estimates[object_name].append(E_stochastic)

            print('Object:', os.path.splitext(file_name)[0])
            print('E_naive:', E_naive)
            print('E_hertz:', E_hertz)
            print('E_MDR:', E_MDR)
            print('E_stochastic:', E_stochastic)
            print('\n')
            
            if PLOT_DATA:
                # Plot raw data
                plotting_label = object_name if file_name.count('t=0') > 0 else '_'
                sp1.plot(estimator.max_depths(), estimator.forces(), ".", label=plotting_label, markersize=8, color=plotting_color)

                # Plot naive fit
                sp2.plot(x_naive, y_naive, ".", label=plotting_label, markersize=8, color=plotting_color)
                sp2.plot(x_naive, E_naive*np.array(x_naive), "-", label=plotting_label, markersize=8, color=plotting_color)

    if PLOT_DATA:
        fig1.legend()
        fig1.set_figwidth(10)
        fig1.set_figheight(10)
        fig2.legend()
        fig2.set_figwidth(10)
        fig2.set_figheight(10)
        plt.show()
        print('Done.')

    # # Find a linear scaling for each set of predictions to minimize error
    # naive_scaling       = scale_predictions(naive_estimates, object_to_modulus)
    # hertz_scaling       = scale_predictions(hertz_estimates, object_to_modulus)
    # MDR_scaling         = scale_predictions(MDR_estimates, object_to_modulus)
    # stochastic_scaling  = scale_predictions(stochastic_estimates, object_to_modulus)
    
    # with open(f'{DATA_DIR}/evaluate_estimator/max_depths.json', 'w') as json_file:
    #     json.dump(max_depths, json_file)

    # print('All skipped files:', skipped_files)

    # # Compute average loss / average log difference / log accuracy for each
    # naive_stats         = compute_estimation_stats(naive_estimates, naive_scaling, object_to_modulus)
    # hertz_stats         = compute_estimation_stats(hertz_estimates, hertz_scaling, object_to_modulus)
    # MDR_stats           = compute_estimation_stats(MDR_estimates, MDR_scaling, object_to_modulus)
    # stochastic_stats    = compute_estimation_stats(stochastic_estimates, stochastic_scaling, object_to_modulus)

    # # Create path to save all generate data in
    # if not os.path.exists(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}'):
    #     os.mkdir(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}')

    # # Save run data for each method
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/naive_config.json', 'w') as json_file:
    #     json.dump(naive_config, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/naive_estimates.json', 'w') as json_file:
    #     json.dump(naive_estimates, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/naive_stats.json', 'w') as json_file:
    #     json.dump(naive_stats, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/hertz_config.json', 'w') as json_file:
    #     json.dump(hertz_config, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/hertz_estimates.json', 'w') as json_file:
    #     json.dump(hertz_estimates, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/hertz_stats.json', 'w') as json_file:
    #     json.dump(hertz_stats, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/MDR_config.json', 'w') as json_file:
    #     json.dump(MDR_config, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/MDR_estimates.json', 'w') as json_file:
    #     json.dump(MDR_estimates, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/MDR_stats.json', 'w') as json_file:
    #     json.dump(MDR_stats, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/stochastic_config.json', 'w') as json_file:
    #     json.dump(stochastic_config, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/stochastic_estimates.json', 'w') as json_file:
    #     json.dump(stochastic_estimates, json_file)
    # with open(f'{DATA_DIR}/evaluate_estimator/{RUN_NAME}/stochastic_stats.json', 'w') as json_file:
    #     json.dump(stochastic_stats, json_file)

    # print('NAIVE CONFIG:\n', naive_config)
    # print('NAIVE METHOD:\n', naive_stats, '\n')
    # print('HERTZ CONFIG:\n', hertz_config)
    # print('HERTZ METHOD:\n', hertz_stats, '\n')
    # print('MDR CONFIG:\n', MDR_config)
    # print('MDR METHOD:\n', MDR_stats, '\n')
    # print('STOCHASTIC CONFIG:\n', stochastic_config)
    # print('STOCHASTIC METHOD:\n', stochastic_stats, '\n')

    # # Create plots showing how well each method does
    # compute_estimation_stats('Naive Elasticity Method', naive_estimates, naive_scaling, object_to_modulus)
    # compute_estimation_stats('Hertzian Method', hertz_estimates, hertz_scaling, object_to_modulus)
    # compute_estimation_stats('MDR', MDR_estimates, MDR_scaling, object_to_modulus)
    # compute_estimation_stats('Stochastic Method', stochastic_estimates, stochastic_scaling, object_to_modulus)