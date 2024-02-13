import os
import csv
import json
import pickle
from tqdm import tqdm
from tactile_estimate import *

wedge_video         = GelsightWedgeVideo(config_csv="./wedge_config/config_100.csv") # Force-sensing finger
other_wedge_video   = GelsightWedgeVideo(config_csv="./wedge_config/config_200_markers.csv") # Non-sensing finger
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

# Define settings configuration for each estimator method
naive_configs = [{
        'contact_mask': None,
        'use_mean': False,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': False,
        'use_lower_resolution_depth': False,
    },{
        'contact_mask': None,
        'use_mean': False,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': False,
        'use_lower_resolution_depth': True,
    },
    {
        'contact_mask': None,
        'use_mean': True,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': False,
        'use_lower_resolution_depth': True,
    },
    {
        'contact_mask': None,
        'use_mean': False,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': True,
        'use_lower_resolution_depth': True,
    },
    {
        'contact_mask': None,
        'use_mean': True,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': True,
        'use_lower_resolution_depth': True,
    }
]
hertz_configs = [{
        'contact_mask': None,
        'use_ellipse_mask': False, 
        'fit_mask_to_ellipse': False,
        'use_lower_resolution_depth': False,
    },{
        'contact_mask': None,
        'use_ellipse_mask': False, 
        'fit_mask_to_ellipse': False,
        'use_lower_resolution_depth': True,
    },{
        'contact_mask': None,
        'use_ellipse_mask': False, 
        'fit_mask_to_ellipse': True,
        'use_lower_resolution_depth': True,
    }
]
MDR_configs = [{
        'contact_mask': None,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': False,
        'use_apparent_deformation': True,
        'use_lower_resolution_depth': True,
        'use_mean_radius': False
    },{
        'contact_mask': None,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': False,
        'use_apparent_deformation': False,
        'use_lower_resolution_depth': True,
        'use_mean_radius': False
    },{
        'contact_mask': None,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': False,
        'use_apparent_deformation': True,
        'use_lower_resolution_depth': False,
        'use_mean_radius': False
    },{
        'contact_mask': None,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': False,
        'use_apparent_deformation': False,
        'use_lower_resolution_depth': False,
        'use_mean_radius': False
    },{
        'contact_mask': None,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': True,
        'use_apparent_deformation': True,
        'use_lower_resolution_depth': True,
        'use_mean_radius': False
    },{
        'contact_mask': None,
        'use_ellipse_mask': False,
        'fit_mask_to_ellipse': False,
        'use_apparent_deformation': False,
        'use_lower_resolution_depth': True,
        'use_mean_radius': True
    }
]

estimator = EstimateModulus(grasp_data=grasp_data, use_gripper_width=True, use_other_video=USE_MARKER_FINGER)

CONTACT_MASKS = ['constant_threshold_contact_mask', 'total_conditional_contact_mask', 'normalized_threshold_contact_mask' ] # , \
                #  'total_normalized_threshold_contact_mask', 'mean_threshold_contact_mask', 'total_mean_threshold_contact_mask', \
                #  'std_above_mean_contact_mask'
                # ]

max_depths = {}

objects = sorted(os.listdir(f'{DATA_DIR}/raw_data'))
for object_name in tqdm(objects):
    data_files = sorted(os.listdir(f'{DATA_DIR}/raw_data/{object_name}'))
    for file_name in data_files:
        if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_other") > 0:
            continue

        trial_number = eval(os.path.splitext(file_name)[0][file_name.find('t=')+2:])
        file_path_prefix = f'{DATA_DIR}/estimations/{object_name}/t={str(trial_number)}'
        if os.path.exists(file_path_prefix):
            if not os.path.exists(f'{DATA_DIR}/estimations/{object_name}'):
                os.mkdir(f'{DATA_DIR}/estimations/{object_name}')
            if not os.path.exists(file_path_prefix):
                os.mkdir(file_path_prefix)

        # Load data into estimator
        estimator._reset_data()
        estimator.load_from_file(f"{DATA_DIR}/raw_data/{object_name}/{os.path.splitext(file_name)[0]}", auto_clip=False)

        # Clip to loading sequence
        estimator.clip_to_press()
        assert len(estimator.depth_images()) == len(estimator.forces()) == len(estimator.gripper_widths())

        # Save maximum depths
        max_depths[os.path.splitext(file_name)[0]] = (int(np.argmax(estimator.max_depths())), estimator.max_depths().max(), -1, estimator.max_depths()[-1])

        # Remove stagnant gripper values across measurement frames
        estimator.interpolate_gripper_widths()
        
        # Loop through all desired contact masks to get estimations
        for contact_mask in CONTACT_MASKS:

            # Fit using naive estimator
            for i in range(len(naive_configs)):
                naive_config = naive_configs[i]
                E_naive = estimator.fit_modulus_naive(
                            contact_mask=contact_mask,
                            use_mean=naive_config['use_mean'],
                            use_ellipse_mask=naive_config['use_ellipse_mask'],
                            fit_mask_to_ellipse=naive_config['fit_mask_to_ellipse'],
                            use_lower_resolution_depth=naive_config['use_lower_resolution_depth']
                        )
                config_contact_mask = naive_config['contact_mask'] if naive_config['contact_mask'] is not None else 'ellipse_contact_mask'
                if not os.path.exists(f'{file_path_prefix}/naive'):
                    os.mkdir(f'{file_path_prefix}/naive')
                if not os.path.exists(f'{file_path_prefix}/naive/{config_contact_mask}'):
                    os.mkdir(f'{file_path_prefix}/naive/{config_contact_mask}')
                with open(f'{file_path_prefix}/naive/{config_contact_mask}/{i}.pkl', 'wb') as file:
                    pickle.dump(E_naive, file)
                with open(f'{file_path_prefix}/naive/{config_contact_mask}/{i}.json', 'w') as json_file:
                    json.dump(naive_config, json_file)

            # Fit using Hertzian estimator
            for i in range(len(hertz_configs)):
                hertz_config = hertz_configs[i]
                E_hertz = estimator.fit_modulus_hertz(
                            contact_mask=contact_mask,
                            use_ellipse_mask=hertz_config['use_ellipse_mask'],
                            fit_mask_to_ellipse=hertz_config['fit_mask_to_ellipse'],
                            use_lower_resolution_depth=hertz_config['use_lower_resolution_depth']
                        )
                config_contact_mask = hertz_config['contact_mask'] if hertz_config['contact_mask'] is not None else 'ellipse_contact_mask'
                if not os.path.exists(f'{file_path_prefix}/hertz'):
                    os.mkdir(f'{file_path_prefix}/hertz')
                if not os.path.exists(f'{file_path_prefix}/hertz/{config_contact_mask}'):
                    os.mkdir(f'{file_path_prefix}/hertz/{config_contact_mask}')
                with open(f'{file_path_prefix}/hertz/{config_contact_mask}/{i}.pkl', 'wb') as file:
                    pickle.dump(E_hertz, file)
                with open(f'{file_path_prefix}/hertz/{config_contact_mask}/{i}.json', 'w') as json_file:
                    json.dump(hertz_config, json_file)

            # Fit using MDR estimator
            for i in range(len(MDR_configs)):
                MDR_config = MDR_configs[i]
                E_MDR = estimator.fit_modulus_MDR(
                                        contact_mask=contact_mask,
                                        use_ellipse_mask=MDR_config['use_ellipse_mask'],
                                        fit_mask_to_ellipse=MDR_config['fit_mask_to_ellipse'],
                                        use_apparent_deformation=MDR_config['use_apparent_deformation'],
                                        use_lower_resolution_depth=MDR_config['use_lower_resolution_depth'],
                                        use_mean_radius=MDR_config['use_mean_radius'],
                                    )
                config_contact_mask = MDR_config['contact_mask'] if MDR_config['contact_mask'] is not None else 'ellipse_contact_mask'
                if not os.path.exists(f'{file_path_prefix}/MDR'):
                    os.mkdir(f'{file_path_prefix}/MDR')
                if not os.path.exists(f'{file_path_prefix}/MDR/{config_contact_mask}'):
                    os.mkdir(f'{file_path_prefix}/MDR/{config_contact_mask}')
                with open(f'{file_path_prefix}/MDR/{config_contact_mask}/{i}.pkl', 'wb') as file:
                    pickle.dump(E_MDR, file)
                with open(f'{file_path_prefix}/MDR/{config_contact_mask}/{i}.json', 'w') as json_file:
                    json.dump(MDR_config, json_file)

            # Fit using the stochastic estimator
            E_stochastic = estimator.fit_modulus_stochastic()
            if not os.path.exists(f'{file_path_prefix}/stochastic'):
                os.mkdir(f'{file_path_prefix}/stochastic')
            with open(f'{file_path_prefix}/stochastic/0.pkl', 'wb') as file:
                pickle.dump(E_stochastic, file)
        
with open(f'{DATA_DIR}/estimations/max_depths.json', 'w') as json_file:
    json.dump(max_depths, json_file)