import os
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

USE_MARKER_FINGER = True

if __name__ == '__main__':

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

    objects = sorted(os.listdir('./data'))
    for object_name in objects:
        plotting_color = random_hex_color()

        # if object_name.count('golf') == 0: continue
        shifts = []

        data_files = sorted(os.listdir(f'./data/{object_name}'))
        for file_name in data_files:
            if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_other") > 0:
                continue

            print(f'Working on {os.path.splitext(file_name)[0]}...')

            # Load data into estimator
            grasp_data._reset_data()
            estimator = EstimateModulus(grasp_data=grasp_data, use_gripper_width=True, use_other_video=USE_MARKER_FINGER)
            estimator.load_from_file(f"./data/{object_name}/{os.path.splitext(file_name)[0]}", auto_clip=False)

            # estimator.watch_depth_2D()

            # estimator.grasp_data.plot_grasp_data()

            forces      = abs(estimator.grasp_data.forces())
            widths      = estimator.grasp_data.gripper_widths()
            max_depths  = estimator.grasp_data.max_depths()
            other_max_depths = estimator.grasp_data.max_depths(other_finger=True)


            # plt.ion()
            # _, ax = plt.subplots(1,2)
            # im = ax[0].imshow(estimator.grasp_data.wedge_video.diff_images()[0], cmap=plt.cm.winter)
            # im1 = ax[1].imshow(estimator.grasp_data.other_wedge_video.diff_images()[0], cmap=plt.cm.winter)
            # for i in range(len(estimator.grasp_data.other_wedge_video.diff_images())):
            #     im.set_array(estimator.grasp_data.wedge_video.diff_images()[i])
            #     im1.set_array(estimator.grasp_data.other_wedge_video.diff_images()[i])
            #     plt.draw()
            #     plt.pause(0.5)
            # plt.ioff()
            # plt.show()

            plt.figure()
            plt.plot(forces / forces.max(), label="Normalized Contact Forces")
            plt.plot(widths / widths.max(), label="Normalized Gripper Widths")
            plt.plot(max_depths / max_depths.max(), label="Normalized Max Depths")
            plt.plot(other_max_depths / other_max_depths.max(), label="Other Normalized Max Depths")
            plt.xlabel('Index [/]')
            plt.title('Unshifted Grasp')
            plt.legend()

            plt.figure()
            plt.plot(forces[8:] / forces.max(), label="Normalized Contact Forces")
            plt.plot(widths[8:] / widths.max(), label="Normalized Gripper Widths")
            plt.plot(max_depths[8:] / max_depths.max(), label="Normalized Max Depths")
            plt.plot(other_max_depths / other_max_depths.max(), label="Other Normalized Max Depths")
            plt.xlabel('Index [/]')
            plt.title('Dumb Shifted Grasp')
            plt.legend()

            shift1 = np.argmax(np.correlate(max_depths / max_depths.max(), other_max_depths / other_max_depths.max(), mode="full")) - len(max_depths) + 1
            other_shifted_depths = np.roll(other_max_depths, shift1)
            print(shift1)
            shifts.append(shift1)

            plt.figure()
            plt.plot(forces / forces.max(), label="Normalized Contact Forces")
            plt.plot(widths / widths.max(), label="Normalized Gripper Widths")
            plt.plot(max_depths / max_depths.max(), label="Normalized Max Depths")
            plt.plot(other_shifted_depths / other_shifted_depths.max(), label="Other Normalized Max Depths")
            plt.xlabel('Index [/]')
            plt.title('Smart Shifted Grasp')
            plt.legend()
            plt.show()

            # plt.ion()
            # _, ax = plt.subplots(1,2)
            # im = ax[0].imshow(estimator.grasp_data.wedge_video.diff_images()[0], cmap=plt.cm.winter)
            # im1 = ax[1].imshow(estimator.grasp_data.other_wedge_video.diff_images()[0], cmap=plt.cm.winter)
            # for i in range(len(estimator.grasp_data.other_wedge_video.diff_images())):
            #     im.set_array(estimator.grasp_data.wedge_video.diff_images()[i])
            #     im1.set_array(estimator.grasp_data.other_wedge_video.diff_images()[i - shift1])
            #     plt.draw()
            #     plt.pause(0.5)
            # plt.ioff()
            # plt.show()

            print('here')
            '''
            # Clip to loading sequence
            estimator.clip_to_press()
            assert len(estimator.depth_images()) == len(estimator.forces()) == len(estimator.gripper_widths())

            # Remove stagnant gripper values across measurement frames
            estimator.interpolate_gripper_widths()

            # Fit using naive estimator
            E_object = estimator.fit_modulus_naive(use_mean=False, use_ellipse_mask=False, fit_mask_to_ellipse=True, use_lower_resolution_depth=True)

            if file_name.count('t=0') > 0:
                plotting_label = object_name
            else:
                plotting_label = '_'

            # Plot raw data
            sp1.plot(estimator.max_depths(), estimator.forces(), ".", label=plotting_label, markersize=8, color=plotting_color)

            # Plot naive fit
            sp2.plot(estimator._x_data, estimator._y_data, ".", label=plotting_label, markersize=8, color=plotting_color)
            sp2.plot(estimator._x_data, E_object*np.array(estimator._x_data), "-", label=plotting_label, markersize=8, color=plotting_color)
            '''

    fig1.legend()
    fig1.set_figwidth(10)
    fig1.set_figheight(10)
    fig2.legend()
    fig2.set_figwidth(10)
    fig2.set_figheight(10)
    plt.show()
    print('Done.')