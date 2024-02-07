import os
import csv
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

USE_MARKER_FINGER = False
PLOT = False

# Objects to exclude from evaluation
EXCLUDE = ['playdoh', 'silly_putty', 'racquetball']

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

if __name__ == '__main__':

    # Read CSV files with objects and labels tabulated
    object_to_modulus = {}
    csv_file_path = f'{DATA_DIR}/objects_and_labels.csv'
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) # Skip title row
        for row in csv_reader:
            if row[14] != '':
                object_to_modulus[row[1]] = float(row[14])

    if PLOT:
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
        'fit_mask_to_ellipse': True,
        'use_lower_resolution_depth': True,
    }
    hertz_config = {
        'contact_mask': None,
        'use_ellipse_mask': True, 
        'fit_mask_to_ellipse': False,
        'use_lower_resolution_depth': False,
    }
    MDR_config = {
        'contact_mask': None,
        'use_ellipse_mask': True,
        'fit_mask_to_ellipse': False,
        'use_apparent_deformation': True,
        'use_lower_resolution_depth': False,
        'use_mean_radius': False
    }
    stochastic_config = {}

    naive_estimates         = {}
    hertz_estimates         = {}
    MDR_estimates           = {}
    stochastic_estimates    = {}

    objects = sorted(os.listdir(f'{DATA_DIR}/raw_data'))
    for object_name in objects:
        if PLOT: plotting_color = random_hex_color()

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
            grasp_data._reset_data()
            estimator = EstimateModulus(grasp_data=grasp_data, use_gripper_width=True, use_other_video=USE_MARKER_FINGER)
            estimator.load_from_file(f"{DATA_DIR}/raw_data/{object_name}/{os.path.splitext(file_name)[0]}", auto_clip=False)

            # Clip to loading sequence
            estimator.clip_to_press()
            assert len(estimator.depth_images()) == len(estimator.forces()) == len(estimator.gripper_widths())

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

            if E_naive < 0:
                print('Negative modulus!')

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
            
            if PLOT:
                # Plot raw data
                plotting_label = object_name if file_name.count('t=0') > 0 else '_'
                sp1.plot(estimator.max_depths(), estimator.forces(), ".", label=plotting_label, markersize=8, color=plotting_color)

                # Plot naive fit
                sp2.plot(x_naive, y_naive, ".", label=plotting_label, markersize=8, color=plotting_color)
                sp2.plot(x_naive, E_naive*np.array(x_naive), "-", label=plotting_label, markersize=8, color=plotting_color)

    # Find a linear scaling for each set of predictions to minimize error
    naive_scaling       = scale_predictions(naive_estimates, object_to_modulus)
    hertz_scaling       = scale_predictions(hertz_estimates, object_to_modulus)
    MDR_scaling         = scale_predictions(MDR_estimates, object_to_modulus)
    stochastic_scaling  = scale_predictions(stochastic_estimates, object_to_modulus)

    # Compute average loss / average log difference / log accuracy for each
    print('NAIVE CONFIG:\n', naive_config)
    print('NAIVE METHOD:\n', compute_estimation_stats(naive_estimates, object_to_modulus), '\n')
    print('HERTZ CONFIG:\n', hertz_config)
    print('HERTZ METHOD:\n', compute_estimation_stats(hertz_estimates, object_to_modulus), '\n')
    print('MDR CONFIG:\n', MDR_config)
    print('MDR METHOD:\n', compute_estimation_stats(MDR_estimates, object_to_modulus), '\n')
    print('STOCHASTIC CONFIG:\n', stochastic_config)
    print('STOCHASTIC METHOD:\n', compute_estimation_stats(stochastic_estimates, object_to_modulus), '\n')

    if PLOT:
        fig1.legend()
        fig1.set_figwidth(10)
        fig1.set_figheight(10)
        fig2.legend()
        fig2.set_figwidth(10)
        fig2.set_figheight(10)
        plt.show()
        print('Done.')