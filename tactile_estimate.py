import os
import cv2
import colorsys
import random
import webcolors
import numpy as np
import warnings
import matplotlib.pyplot as plt

from wedge_video import GelsightWedgeVideo, DEPTH_THRESHOLD
from contact_force import ContactForce, FORCE_THRESHOLD
from gripper_width import GripperWidth, SMOOTHING_POLY_ORDER
from grasp_data import GraspData

grasp_data = GraspData()

from scipy.ndimage import convolve

# Archived measurements from calipers on surface in x and y directions
WARPED_PX_TO_MM = (11, 11)
RAW_PX_TO_MM = (12.5, 11)

# Use sensor size and warped data to choose conversion
IMG_R, IMG_C = 300, 400
SENSOR_PAD_DIM_MM = (24, 33) # [mm]
PX_TO_MM = np.sqrt((IMG_R / SENSOR_PAD_DIM_MM[0])**2 + (IMG_C / SENSOR_PAD_DIM_MM[1])**2)
MM_TO_PX = 1/PX_TO_MM

# Random shades for consistent plotting over multiple trials
def random_shade_of_color(color_name):
    try:
        rgb = webcolors.name_to_rgb(color_name)
        hls = colorsys.rgb_to_hls(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)

        # Randomize the lightness while keeping hue and saturation constant
        lightness = random.uniform(0.5, 1.0)
        rgb_shaded = colorsys.hls_to_rgb(hls[0], lightness, hls[2])
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb_shaded[0] * 255),
            int(rgb_shaded[1] * 255),
            int(rgb_shaded[2] * 255)
        )
        return hex_color

    except ValueError:
        raise ValueError("Invalid color name")

class EstimateModulus():
    def __init__(
            self, depth_threshold=0.001*DEPTH_THRESHOLD, force_threshold=FORCE_THRESHOLD, 
            assumed_poissons_ratio=0.45, use_gripper_width=True
        ):

        self.assumed_poisson_ratio = assumed_poissons_ratio # [\]
        self.depth_threshold = depth_threshold # [m]
        self.force_threshold = force_threshold # [N]

        self.use_gripper_width = use_gripper_width # Boolean of whether or not to include gripper width
        self.grasp_data = GraspData(use_gripper_width=self.use_gripper_width)

        # Gel material: Silicone XP-565
        # Datasheet:
        #     "https://static1.squarespace.com/static/5b1ecdd4372b96e84200cf1d/t/5b608e1a758d46a989c0bf93/1533054492488/XP-565+%281%29.pdf"
        # Estimated from Shore 00 hardness:
        #     "https://www.dow.com/content/dam/dcc/documents/en-us/tech-art/11/11-37/11-3716-01-durometer-hardness-for-silicones.pdf"
        self.shore_00_hardness = 60 # From durometer measurement
        self.E_gel = 275000 # N/m^2
        self.nu_gel = 0.485 # [\]
        self.gel_width = 0.035 # [m]
        self.gel_depth = 0.0055 # [m] (slightly adjusted because of depth inaccuracies)

        self._F = []                 # Contact forces for fitting
        self._d = []                 # Contact depth for fitting
        self._a = []                 # Contact radius for fitting
        self._contact_areas = []     # Contact areas for fitting
        self._x_data = []            # Save fitting data for plotting
        self._y_data = []            # Save fitting data for plotting

    # Clear out the data values
    def _reset_data(self):
        self.grasp_data = GraspData(use_gripper_width=self.use_gripper_width)
        self._F = []
        self._d = []
        self._a = []
        self._contact_areas = []
        self._x_data = []
        self._y_data = []

    # Load data from a file
    def load_from_file(self, path_to_file, auto_clip=True):
        self._reset_data()
        self.grasp_data.load(path_to_file)
        if auto_clip: # Clip to the entire press
            self.grasp_data.auto_clip()
        assert len(self.grasp_data.depth_images()) == len(self.grasp_data.forces())
        return

    # Return forces
    def forces(self):
        return self.grasp_data.forces()
    
    # Return gripper widths
    def gripper_widths(self):
        assert self.use_gripper_width
        return self.grasp_data.gripper_widths()
    
    # Return depth images (in meters)
    def depth_images(self):
        return 0.001 * self.grasp_data.depth_images()
    
    # Return maximum value from each depth image (in meters)
    def max_depths(self):
        return 0.001 * self.grasp_data.max_depths()
    
    # Return mean value from each depth image (in meters)
    def mean_depths(self):
        return 0.001 * self.grasp_data.mean_depths()
    
    # Return mean of neighborhood around max value from each depth image (in meters)
    def mean_max_depths(self, kernel_radius=5):
        mean_max_depths = []
        for i in range(len(self.depth_images())):
            depth_image = self.depth_images()[i]
            max_index = np.argmax(depth_image)
            r, c = np.unravel_index(max_index, depth_image.shape)
            mean_max_depth = depth_image[r-kernel_radius:r+kernel_radius, c-kernel_radius:c+kernel_radius].mean()
            mean_max_depths.append(mean_max_depth)
        return np.array(mean_max_depths)
    
    # Return highest percentile of depth population (in meters)
    def top_percentile_depths(self, percentile=97):
        top_percentile_depths = []
        for i in range(len(self.depth_images())):
            top_percentile_depths.append(np.percentile(self.depth_images()[i], percentile))
        return np.array(top_percentile_depths)
    
    # Clip a press sequence to only the loading sequence (positive force)
    def clip_to_press(self, use_force=True):
        # Find maximum depth over press
        if use_force:
            i_start = np.argmax(self.forces() >= self.force_threshold)
            i_peak = np.argmax(self.forces())

            # Grab index before below 97.5% of peak
            i_end = i_peak
            for i in range(len(self.forces())):
                if i > i_peak and self.forces()[i] <= 0.975*self.forces()[i_peak]:
                    i_end = i-1
                    break
        else:
            # Find peak and start over depth values
            i_start = np.argmax(self.max_depths() >= self.depth_threshold)
            i_peak = np.argmax(self.max_depths())
            i_end = i_peak

        if i_start >= i_end:
            warnings.warn("No press detected! Cannot clip.", Warning)
        else:
            # Clip from start to peak depth
            self.grasp_data.clip(i_start, i_end+1)
        return

    # Return mask of which pixels are in contact with object
    def contact_mask(self, depth):
        return depth >= self.depth_threshold
    
    # Return mask of only where depth is concave
    def concave_mask(self, depth):
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        laplacian = convolve(depth, laplacian_kernel, mode='constant', cval=0)

        # Create a mask: 1 for convex regions, 0 for concave regions
        return (laplacian <= 0).astype(int)

    # Return mask to disgard outside pixels
    # Vertically shift crop away from bottom (where depth is most noisy)
    def crop_edges(self, depth, vertical_shift=CROP_VERTICAL_SHIFT):
        assert depth.shape[0] > 2*self.edge_crop_margin and depth.shape[1] > 2*self.edge_crop_margin
        filtered_depth = depth.copy()
        filtered_depth[0:self.edge_crop_margin, :] = 0
        filtered_depth[:, 0:self.edge_crop_margin+vertical_shift] = 0
        filtered_depth[depth.shape[0]-self.edge_crop_margin:depth.shape[0], :] = 0
        filtered_depth[:, depth.shape[1]-self.edge_crop_margin+vertical_shift:depth.shape[1]] = 0
        return filtered_depth
    
    # Filter all depth images using masks and cropping
    def filter_depths(self, threshold_contact=False, concave_mask=False):
        for i in range(self.depth_images().shape[0]):
            filtered_depth = self.depth_images()[i,:,:].copy()

            # Mask depth to consider contact area only
            if threshold_contact:
                contact_mask = self.contact_mask(filtered_depth)
                filtered_depth = filtered_depth * contact_mask

            if concave_mask: # Only consider convex points on surface
                filtered_depth = self.concave_mask(filtered_depth) * filtered_depth
            self.grasp_data.wedge_video._depth_images[i] = filtered_depth

        return

    # Fit to continuous function and down sample to smooth measurements
    def smooth_gripper_widths(self, plot_smoothing=False, poly_order=SMOOTHING_POLY_ORDER):
        self.grasp_data.gripper_width.smooth_gripper_widths(plot_smoothing=plot_smoothing, poly_order=poly_order)
        return

    # Fit linear equation with least squares
    def linear_coeff_fit(self, x, y):
        # Solve for best A given data and equation of form y = A*x
        return np.dot(x, y) / np.dot(x, x)
    
    # Convert depth image to 3D data
    def depth_to_XYZ(self, depth, remove_zeros=True, remove_outliers=True):
        # Extract data
        X, Y, Z = [], [], []
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                if (not remove_zeros) or (remove_zeros and depth[i][j] >= 1e-9):
                    X.append(0.001 * i / PX_TO_MM) # Convert to meters
                    Y.append(0.001 * j / PX_TO_MM)
                    Z.append(depth[i][j])
        data = np.vstack((np.array(X), np.array(Y), np.array(Z)))   
        
        # Remove outliers via ~3-sigma rule
        if remove_outliers == True:
            X, Y, Z = [], [], []
            centroid = np.mean(data, axis=1)
            sigma = np.std(data, axis=1)
            for i in range(data.shape[1]):
                point = data[:, i]
                if (np.abs(centroid - point) <= 3*sigma).all():
                    X.append(point[0])
                    Y.append(point[1])
                    Z.append(point[2])

        # TODO: Add (k-means) clustering to pick out point? (if necessary)

        return np.array(X), np.array(Y), np.array(Z)

    # Fit depth points to a sphere in 3D space to get contact depth and radius
    def fit_depth_to_sphere(self, depth, min_datapoints=10):
        '''
        Modified approach from: https://jekel.me/2015/Least-Squares-Sphere-Fit/
        '''
        # Extract data
        X, Y, Z = self.depth_to_XYZ(depth)
        if X.shape[0] < min_datapoints:
            return [0, 0, 0, 0]

        # Construct data matrix
        A = np.zeros((len(X),4))
        A[:,0] = X*2
        A[:,1] = Y*2
        A[:,2] = Z*2
        A[:,3] = 1

        # Assemble the f matrix
        f = np.zeros((len(X),1))
        f[:,0] = (X*X) + (Y*Y) + (Z*Z)

        # Solve least squares for sphere
        C, res, rank, s = np.linalg.lstsq(A, f, rcond=None)

        # Solve for the radius
        radius = np.sqrt((C[0]*C[0]) + (C[1]*C[1]) + (C[2]*C[2]) + C[3])

        return [radius, C[0], C[1], C[2]] # [ radius, center_x, center_y, center_z ]
    
    # Compute radius of contact from sphere fit
    def estimate_contact_radius(self, sphere):
        return np.sqrt(max(sphere[0]**2 - sphere[3]**2, 0))

    # Compute depth of contact from sphere fit
    def estimate_contact_depth(self, sphere):
        return sphere[0] + sphere[3]
    
    def fit_modulus_MDR(self):
        # Following MDR algorithm from (2.3.2) in "Handbook of Contact Mechanics" by V.L. Popov

        # p_0     = f(E*, F, a)
        # p(r)    = p_0 sqrt(1 - r^2/a^2)
        # q_1d(x) = 2 integral(r p(r) / sqrt(r^2 - x^2) dr)
        # w_1d(x) = (1-v^2)/E_sensor * q_1d(x)
        # w_1d(0) = max_depth

        # Filter depth images to mask and crop
        self.filter_depths(concave_mask=False)

        x_data, y_data = [], []
        d, a, F = [], [], []

        # TODO: Generalize this radius
        R = 0.025 # [m], measured for elastic balls

        for i in range(self.depth_images().shape[0]):
            F_i = abs(self.forces()[i])
            contact_area = (0.001 / PX_TO_MM)**2 * np.sum(self.depth_images()[i] > 1e-10)
            a_i = np.sqrt(contact_area / np.pi)

            # Take mean of 5x5 neighborhood around maximum depth
            max_index = np.unravel_index(np.argmax(self.depth_images()[i]), self.depth_images()[i].shape)
            d_i = np.mean(self.depth_images()[i][max_index[0]-2:max_index[0]+3, max_index[1]-2:max_index[1]+3])

            if F_i > 0 and a_i >= 0.003 and d_i > self.depth_threshold:
                p_0 = (1/np.pi) * (6*F_i/(R**2))**(1/3) # times E_star^2/3
                q_1D_0 = p_0 * np.pi * a_i / 2
                w_1D_0 = (1 - self.nu_gel**2) * q_1D_0 / self.E_gel
                F.append(F_i)
                d.append(d_i)
                a.append(a_i)
                x_data.append(w_1D_0)
                y_data.append(d_i)

        self._d = np.array(d)
        self._a = np.array(a)
        self._F = np.array(F)
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

        # Fit for E_star
        E_star = self.linear_coeff_fit(x_data, y_data)**(3/2)
        return E_star
    
    def Estar_to_E(self, E_star):
        # Compute compliance from E_star by assuming Poisson's ratio
        nu = self.assumed_poisson_ratio
        E = (1 - nu**2) / (1/E_star - (1 - self.nu_gel**2)/(self.E_gel))
        return E, nu
    
    # Naively estimate modulus based on gripper width change and aggregate modulus
    def fit_modulus_naive(self, use_mean=False):
        assert self.use_gripper_width

        # Find initial length of first contact
        L0 = self.gripper_widths()[0]
        d0 = 0
        
        contact_areas, a = [], []
        x_data, y_data, d = [], [], []
        for i in range(len(self.depth_images())):
            depth_i = self.depth_images()[i]

            if use_mean:
                # d_i = depth_i.mean()
                d_i = depth_i[self.edge_crop_margin:depth_i.shape[0]-self.edge_crop_margin, \
                              self.edge_crop_margin+20:depth_i.shape[1]-self.edge_crop_margin+20].mean()
                contact_area_i = (0.001 / PX_TO_MM)**2 * (depth_i.shape[0] - 2*self.edge_crop_margin) * (depth_i.shape[1] - 2*self.edge_crop_margin)
            else:
                d_i = self.top_percentile_depths()[i] # depth_i.max()
                contact_area_i = (0.001 / PX_TO_MM)**2 * np.sum(depth_i >= self.depth_threshold)

            a_i = np.sqrt(contact_area_i / np.pi)

            dL = -(self.gripper_widths()[i] - L0 + 2*(d_i - d0))
            if contact_area_i >= 1e-5 and dL >= 0:
                x_data.append(dL/L0) # Strain
                y_data.append(abs(self.forces()[i]) / contact_area_i) # Stress
                contact_areas.append(contact_area_i)
                a.append(a_i)
                d.append(d_i)

        # Save stuff for plotting
        self._x_data = x_data
        self._y_data = y_data
        self._contact_areas = contact_areas
        self._a = a
        self._d = d

        # Fit to modulus
        E = self.linear_coeff_fit(x_data, y_data)
        # E = (1/E_agg - 1/self.E_gel)**(-1)

        return E
    
    # Fit to Hertizan model with apparent deformation
    def fit_modulus_hertz(self):
        # Calculate apparent deformation using gripper width
        # Pretend that the contact geometry is cylindrical
        # This gives the relation...
        #       F_N  =  2 E* d a
        #       [From (2.3.2) in "Handbook of Contact Mechanics" by V.L. Popov]

        # Find initial length of first contact
        L0 = self.gripper_widths()[0]

        x_data, y_data = [], []
        d, contact_areas, a = [], [], []
        for i in range(len(self.depth_images())):
            depth_i = self.depth_images()[i]
            d_i = L0 - self.gripper_widths()[i]
            
            # contact_area_i = (0.001 / PX_TO_MM)**2 * (depth_i.shape[0] - 2*self.edge_crop_margin) * (depth_i.shape[1] - 2*self.edge_crop_margin)
            contact_area_i = (0.001 / PX_TO_MM)**2 * np.sum(depth_i >= self.depth_threshold)

            a_i = np.sqrt(contact_area_i / np.pi)

            if contact_area_i >= 1e-5 and d_i > 0:
                x_data.append(2*d_i*a_i)
                y_data.append(self.forces()[i])
                contact_areas.append(contact_area_i)
                d.append(d_i)
                a.append(a_i)

        self._x_data = x_data
        self._y_data = y_data
        self._contact_areas = contact_areas
        self._a = a
        self._d = d

        E_agg = self.linear_coeff_fit(x_data, y_data)
        E = (1/E_agg - 1/self.E_gel)**(-1)  

        return E
    
    # Check sphere fit by plotting data and fit shape
    def plot_depth(self, depth):
        # Extract 3D data
        X, Y, Z = self.depth_to_XYZ(depth, remove_zeros=False, remove_outliers=False)

        # Plot sphere in 3D
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        axes.scatter(X, Y, Z, s=8, c=Z, cmap='winter', rasterized=True)
        axes.set_xlabel('$X$ [m]',fontsize=16)
        axes.set_ylabel('\n$Y$ [m]',fontsize=16)
        axes.set_zlabel('\n$Z$ [m]',fontsize=16)
        axes.set_title('Sphere Fitting',fontsize=16)
        plt.show()
        return
    
    # Check sphere fit by plotting data and fit shape
    def plot_sphere_fit(self, depth, sphere):
        # Extract 3D data
        X, Y, Z = self.depth_to_XYZ(depth)

        # Create discrete graph of sphere mesh
        r, x0, y0, z0 = sphere
        # u, v = np.mgrid[0:2*np.pi:20j, 0:np.cos(-z0/r):10j] # Plot top half of sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi/10:10j] # Plot top half of sphere
        # u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:20j] # Plot full sphere
        sphere_x = x0 + np.cos(u)*np.sin(v)*r
        sphere_y = y0 + np.sin(u)*np.sin(v)*r
        sphere_z = z0 + np.cos(v)*r

        # Plot sphere in 3D
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        axes.scatter(X, Y, Z, s=8, c=Z, cmap='winter', rasterized=True)
        axes.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r")
        axes.set_xlabel('$X$ [m]',fontsize=16)
        axes.set_ylabel('\n$Y$ [m]',fontsize=16)
        axes.set_zlabel('\n$Z$ [m]',fontsize=16)
        axes.set_title('Sphere Fitting',fontsize=16)
        plt.show()
        return
    
    # Plot different ways of aggregating depth from each image
    def plot_depth_metrics(self):
        plt.plot(self.max_depths(), label="Max Depth")
        plt.plot(self.mean_max_depths(), label="Mean Max Depth")
        plt.plot(self.top_percentile_depths(), label="Top Percentile Depth")
        plt.plot(self.mean_depths(), label="Mean Depth")
        plt.xlabel('Index [/]')
        plt.ylabel('Depth [m]')
        plt.legend()
        plt.show()
        return
    
    # Plot all data over indices
    def plot_grasp_data(self):
        plt.plot(abs(self.forces()) / abs(self.forces()).max(), label="Normalized Forces")
        plt.plot(self.gripper_widths() / self.gripper_widths().max(), label="Normalized Gripper Width")
        plt.plot(self.max_depths() / self.max_depths().max(), label="Normalized Depth")
        plt.legend()
        plt.show()
        return


if __name__ == "__main__":

    ##################################################
    # GET ESTIMATED MODULUS (E) FOR SET OF TEST DATA #
    ##################################################

    fig1 = plt.figure(1)
    sp1 = fig1.add_subplot(211)
    sp1.set_xlabel('Measured Sensor Deformation (d) [m]')
    sp1.set_ylabel('Force [N]')
    
    fig2 = plt.figure(2)
    sp2 = fig2.add_subplot(211)
    sp2.set_xlabel('dL / L')
    sp2.set_ylabel('F / A')

    wedge_video    = GelsightWedgeVideo(config_csv="./config_100.csv") # Force-sensing finger
    contact_force  = ContactForce()
    gripper_width  = GripperWidth()
    grasp_data     = GraspData(wedge_video=wedge_video, contact_force=contact_force, gripper_width=gripper_width, use_gripper_width=True)

    # For plotting
    obj_to_color = {
        "yellow_foam_brick_softest" : "yellow",
        "red_foam_brick_softer"     : "red",
        "blue_foam_brick_harder"    : "blue",
        "orange_ball_softest"       : "orange",
        "green_ball_softer"         : "green",
        "purple_ball_hardest"       : "indigo",
        "rigid_strawberry"          : "purple",
        "golf_ball"                 : "gray",
    }

    max_uncontacted_depths = []
    mean_uncontacted_depths = []
    std_uncontacted_depths = []

    # Unload data from folder
    data_folder = "./example_data/2023-12-16"
    data_files = os.listdir(data_folder)
    for i in range(len(data_files)):
        file_name = data_files[i]
        if os.path.splitext(file_name)[1] != '.avi':
            continue
        obj_name = os.path.splitext(file_name)[0].split('__')[0]

        if file_name[0] != 'y': continue

        # Load data and clip
        estimator = EstimateModulus(use_gripper_width=True)
        estimator.load_from_file(data_folder + "/" + os.path.splitext(file_name)[0], auto_clip=True)
        
        max_uncontacted_depths.append(estimator.depth_images()[0].max())
        mean_uncontacted_depths.append(estimator.depth_images()[0].mean())
        std_uncontacted_depths.append(np.std(estimator.depth_images()[0]))
        
        estimator.clip_to_press()
        assert len(estimator.depth_images()) == len(estimator.forces()) == len(estimator.gripper_widths())

        # Filter depth data?
        estimator.filter_depths(threshold_contact=False, concave_mask=False, crop_edges=True)

        estimator.plot_depth(estimator.depth_images()[-1])

        print(file_name, estimator.depth_images()[-1][150, 200])
        

    print('here')

    #     # Filter depth data?
    #     estimator.filter_depths(threshold_contact=False, concave_mask=False, crop_edges=True)

    #     estimator.smooth_gripper_widths()

    #     # # Fit using our MDR estimator
    #     # E_star = estimator.fit_modulus()
    #     # E_object, v_object = estimator.Estar_to_E(E_star)

    #     E_object = estimator.fit_modulus_naive()

    #     print(f'Maximum depth of {obj_name}:', np.max(estimator.max_depths()))
    #     print(f'Maximum force of {obj_name}:', np.max(estimator.forces()))
    #     print(f'Strain range of {obj_name}:', min(estimator._x_data), 'to', max(estimator._x_data))
    #     print(f'Stress range of {obj_name}:', min(estimator._y_data), 'to', max(estimator._y_data))
    #     print(f'Contact radius range of {obj_name}:', min(estimator._a), 'to', max(estimator._a))
    #     print(f'Depth range of {obj_name}:', min(estimator._d), 'to', max(estimator._d))
    #     print(f'Estimated modulus of {obj_name}:', E_object)
    #     print('\n')

    #     # Plot
    #     plotting_color = random_shade_of_color(obj_to_color[obj_name])
    #     sp1.plot(estimator.max_depths(), estimator.forces(), ".", label=obj_name, markersize=8, color=plotting_color)
    #     sp2.plot(estimator._x_data, estimator._y_data, ".", label=obj_name, markersize=8, color=plotting_color)

    # fig1.legend()
    # fig1.set_figwidth(10)
    # fig1.set_figheight(10)
    # fig2.legend()
    # fig2.set_figwidth(10)
    # fig2.set_figheight(10)
    # plt.show()