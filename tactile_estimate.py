import os
import cv2
import colorsys
import random
import webcolors
import numpy as np
import warnings
import matplotlib.pyplot as plt

from wedge_video import GelsightWedgeVideo, DEPTH_THRESHOLD, WARPED_IMG_SIZE
from contact_force import ContactForce, FORCE_THRESHOLD
from gripper_width import GripperWidth, SMOOTHING_POLY_ORDER
from grasp_data import GraspData

grasp_data = GraspData()

# Archived measurements from calipers on surface in x and y directions
WARPED_PX_TO_MM = (11, 11)
RAW_PX_TO_MM = (12.5, 11)

# Use sensor size and warped data to choose conversion
SENSOR_PAD_DIM_MM = (24, 33) # [mm]
PX_TO_MM = np.sqrt((WARPED_IMG_SIZE[0] / SENSOR_PAD_DIM_MM[0])**2 + (WARPED_IMG_SIZE[1] / SENSOR_PAD_DIM_MM[1])**2)
MM_TO_PX = 1/PX_TO_MM

# Fit an ellipse bounding the True space of a 2D binary array
def fit_ellipse(binary_array, plot_result=False):
    # Find contours in the binary array
    binary_array_uint8 = binary_array.astype(np.uint8)
    contours, _ = cv2.findContours(binary_array_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('No ellipse found!')

    # Iterate through contours
    max_ellipse_area = 0
    for contour in contours:
        # Fit ellipse to the contour
        ellipse = cv2.fitEllipse(contour)

        # Calculate the area of the fitted ellipse
        ellipse_area = (np.pi * ellipse[1][0] * ellipse[1][1]) / 4

        # Check if the ellipse area is above the minimum threshold
        if ellipse_area > max_ellipse_area:
            max_ellipse_area = ellipse_area
            max_ellipse = ellipse

    if plot_result:
        # Draw the ellipse on a blank image for visualization
        ellipse_image = np.zeros_like(binary_array, dtype=np.uint8)
        cv2.ellipse(ellipse_image, max_ellipse, 255, 1)

        # Display the results
        cv2.imshow("Original Binary Array", (binary_array * 255).astype(np.uint8))
        cv2.imshow("Ellipse Fitted", ellipse_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return max_ellipse

# Fit an ellipse bounding the True space of a 2D non-binary array
def fit_ellipse_float(float_array, plot_result=False):
    # Normalize array
    float_array_normalized = (float_array - float_array.min()) / (float_array.max() - float_array.min())

    # Threshold into binary array based on range
    binary_array = (255 * (float_array_normalized >= 0.5)).astype(np.uint8)

    # Find contours in the array
    contours, _ = cv2.findContours(binary_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('No ellipse found!')

    # Iterate through contours
    max_ellipse_area = 0
    for contour in contours:
        if contour.shape[0] < 5: continue
        # Fit ellipse to the contour
        ellipse = cv2.fitEllipse(contour)

        # Calculate the area of the fitted ellipse
        ellipse_area = (np.pi * ellipse[1][0] * ellipse[1][1]) / 4

        # Check if the ellipse area is above the minimum threshold
        if ellipse_area > max_ellipse_area:
            max_ellipse_area = ellipse_area
            max_ellipse = ellipse

    if plot_result:
        # Draw the ellipse on a blank image for visualization
        ellipse_image = np.zeros_like(float_array, dtype=np.uint8)
        cv2.ellipse(ellipse_image, max_ellipse, 255, 1)

        # Display the results
        cv2.imshow("Normalized Array", float_array_normalized)
        cv2.imshow("Binary Array", binary_array)
        cv2.imshow("Ellipse Fitted", ellipse_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return max_ellipse

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
        self._R = []                 # Estimated radius of object
        self._contact_areas = []     # Contact areas for fitting
        self._x_data = []            # Save fitting data for plotting
        self._y_data = []            # Save fitting data for plotting

    # Clear out the data values
    def _reset_data(self):
        self.grasp_data = GraspData(use_gripper_width=self.use_gripper_width)
        self._F = []
        self._d = []
        self._a = []
        self._R = []
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

    # Return mask of which pixels are in contact with object based on constant threshold
    def constant_threshold_contact_mask(self, depth):
        return depth >= self.depth_threshold

    # Return mask of which pixels are in contact with object based on mean of image
    def mean_threshold_contact_mask(self, depth):
        return depth >= depth.mean()
    
    # Return mask of which pixels are in contact with object based on mean of all images
    def total_mean_threshold_contact_mask(self, depth):
        return depth >= self.mean_depths().mean()

    # Return mask of which pixels are in contact with object based on range of image
    def range_threshold_contact_mask(self, depth):
        halfway = 0.5*(depth.max() - depth.min()) + depth.min()
        return depth >= halfway

    # Return mask of which pixels are in contact with object using constant threshold
    # But, flip if the mean depth of the image is negative (...a bit hacky)
    def flipped_threshold_contact_mask(self, depth):
        mask = depth >= self.depth_threshold
        if depth.mean() < 0:
            mask = depth <= -self.depth_threshold
        return mask
    
    # Same as flipped, but use relative threshold
    def flipped_total_mean_threshold_contact_mask(self, depth):
        mask = depth >= self.mean_depths().mean()
        if self.mean_depths().mean() < 0:
            mask = depth <= -self.mean_depths().mean()
        return mask
    
    # Wrap the chosen contact mask function into one place
    def contact_mask(self, depth):
        return self.flipped_total_mean_threshold_contact_mask(depth)

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
                if (not remove_zeros) or (remove_zeros and depth[i][j] >= 1e-10):
                    X.append(0.001 * i / PX_TO_MM) # Convert pixels to meters
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
    
    # Naively estimate modulus based on gripper width change and aggregate modulus
    def fit_modulus_naive(self, use_mean=True, use_ellipse_fitting=True):
        assert self.use_gripper_width

        # Find initial length of first contact
        L0 = self.gripper_widths()[0]
        d0 = 0
        
        contact_areas, a = [], []
        x_data, y_data, d = [], [], []
        for i in range(len(self.depth_images())):
            depth_i = self.depth_images()[i]

            mask = self.contact_mask(depth_i)

            if use_mean:
                d_i = np.sum(depth_i * mask) / np.sum(mask)
            else:
                d_i = self.top_percentile_depths()[i]

            if use_ellipse_fitting:
                # Compute contact area using ellipse fit
                try:
                    ellipse = fit_ellipse(mask)
                except:
                    continue
                if ellipse is None: continue
                major_axis, minor_axis = ellipse[1]
                contact_area_i = np.pi * major_axis * minor_axis
                a_i = (major_axis + minor_axis) / 2
            else:
                # Use mask for contact area
                contact_area_i = (0.001 / PX_TO_MM)**2 * np.sum(mask)
                a_i = np.sqrt(contact_area_i / np.pi)

            dL = -(self.gripper_widths()[i] - L0 + 2*(d_i - d0))
            if dL >= 0 and contact_area_i >= 2.5e-5:
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
        E = self.linear_coeff_fit(x_data, y_data) # np.polyfit(x_data, y_data, 1)
        # E = (1/E_agg - 1/self.E_gel)**(-1)

        return E
    
    # Fit to Hertizan model with apparent deformation
    def fit_modulus_hertz(self, use_ellipse_fitting=True):
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
            
            mask = self.contact_mask(depth_i)

            if use_ellipse_fitting:
                # Compute contact area using ellipse fit
                try:
                    ellipse = fit_ellipse(mask)
                except:
                    continue
                if ellipse is None: continue
                major_axis, minor_axis = ellipse[1]
                contact_area_i = np.pi * major_axis * minor_axis
                a_i = (major_axis + minor_axis) / 2
            else:
                # Use mask for contact area
                contact_area_i = (0.001 / PX_TO_MM)**2 * np.sum(mask)
                a_i = np.sqrt(contact_area_i / np.pi)

            if contact_area_i >= 2.5e-5 and d_i > 0:
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
    
    # Use Hertzian contact models and MDR to compute modulus
    def fit_modulus_MDR(self, use_ellipse_fitting=True):
        # Following MDR algorithm from (2.3.2) in "Handbook of Contact Mechanics" by V.L. Popov

        # p_0     = f(E*, F, a)
        # p(r)    = p_0 sqrt(1 - r^2/a^2)
        # q_1d(x) = 2 integral(r p(r) / sqrt(r^2 - x^2) dr)
        # w_1d(x) = (1-v^2)/E_sensor * q_1d(x)
        # w_1d(0) = max_depth

        # TODO: Replace this
        # # Filter depth images to mask and crop
        # self.filter_depths(concave_mask=False)

        x_data, y_data = [], []
        d, a, F, R = [], [], [], []

        # TODO: Generalize this radius
        # R = 0.025 # [m], measured for elastic balls

        mean_max_depths = self.mean_max_depths()

        for i in range(len(self.depth_images())):
            F_i = abs(self.forces()[i])
            
            mask = self.contact_mask(self.depth_images()[i])
            contact_area_i = (0.001 / PX_TO_MM)**2 * np.sum(mask)
            a_i = np.sqrt(contact_area_i / np.pi)

            # Take mean of 5x5 neighborhood around maximum depth
            d_i = mean_max_depths[i]

            if use_ellipse_fitting:
                # Compute circle radius using ellipse fit
                try:
                    ellipse = fit_ellipse(mask)
                except:
                    continue
                if ellipse is None: continue
                major_axis, minor_axis = ellipse[1]
                r_i = 0.5 * (0.001 / PX_TO_MM) * (major_axis + minor_axis)/2
                R_i = d_i + (r_i**2 - d_i**2)/(2*d_i)
            else:
                # Compute estimated radius based on depth (d) and contact radius (a)
                R_i = d_i + (a_i**2 - d_i**2)/(2*d_i)

            if F_i > 0 and contact_area_i >= 2.5e-5 and d_i > self.depth_threshold:
                p_0 = (1/np.pi) * (6*F_i/(R_i**2))**(1/3) # times E_star^2/3
                q_1D_0 = p_0 * np.pi * a_i / 2
                w_1D_0 = (1 - self.nu_gel**2) * q_1D_0 / self.E_gel
                F.append(F_i)
                d.append(d_i)
                a.append(a_i)
                R.append(R_i)
                x_data.append(w_1D_0)
                y_data.append(d_i)

        self._d = np.array(d)
        self._a = np.array(a)
        self._F = np.array(F)
        self._R = np.array(R)
        self._x_data = np.array(x_data)
        self._y_data = np.array(y_data)

        # Fit for E_star
        E_star = self.linear_coeff_fit(x_data, y_data)**(3/2)
        return E_star
    
    def Estar_to_E(self, E_star):
        # Compute compliance from E_star by assuming Poisson's ratio
        nu = self.assumed_poisson_ratio
        E = (1 - nu**2) / (1/E_star - (1 - self.nu_gel**2)/(self.E_gel))
        return E, nu

    # Display raw data from a depth image in 3D
    def plot_depth(self, depth):
        # Extract 3D data
        X, Y, Z = self.depth_to_XYZ(depth, remove_zeros=False, remove_outliers=False)

        # Plot sphere in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, s=8, c=Z, cmap='winter', rasterized=True)
        ax.set_xlabel('$X$ [m]', fontsize=16)
        ax.set_ylabel('\n$Y$ [m]', fontsize=16)
        ax.set_zlabel('\n$Z$ [m]', fontsize=16)
        ax.set_title('Sphere Fitting', fontsize=16)
        plt.show()
        return
    
    # Display raw data from a depth image as 2D heightmap
    def plot_depth_2D(self, depth):
        plt.figure()
        plt.imshow(depth, cmap="winter")
        plt.title(f'Depth')
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.colorbar()
        plt.show()
    
    # Watch evolution of depth images over time
    def watch_depth_2D(self):
        plt.ion()
        _, ax = plt.subplots()
        ax.set_title(f'Depth')
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        im = ax.imshow(self.depth_images()[0], cmap="winter")
        for i in range(len(self.depth_images())):
            im.set_array(self.depth_images()[i])
            plt.draw()
            plt.pause(0.5)
        plt.ioff()
        plt.show()
        return
    
    # Display computed contact mask for a given depth image
    def plot_contact_mask(self, depth):
        plt.figure()
        plt.imshow(self.contact_mask(depth), cmap=plt.cm.gray)
        plt.title(f'Contact Mask')
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.colorbar()
        plt.show()
        return
    
    # Watch evolution of computed contact mask over time
    def watch_contact_mask(self):
        plt.ion()
        _, ax = plt.subplots()
        ax.set_title(f'Contact Mask')
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        im = ax.imshow(self.contact_mask(self.depth_images()[0]), cmap=plt.cm.gray)
        for i in range(len(self.depth_images())):
            im.set_array(self.contact_mask(self.depth_images()[i]))
            plt.draw()
            plt.pause(0.5)
        plt.ioff()
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
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, s=8, c=Z, cmap='winter', rasterized=True)
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r")
        ax.set_xlabel('$X$ [m]', fontsize=16)
        ax.set_ylabel('\n$Y$ [m]', fontsize=16)
        ax.set_zlabel('\n$Z$ [m]', fontsize=16)
        ax.set_title('Sphere Fitting', fontsize=16)
        plt.show()
        return
    
    # Plot different ways of aggregating depth from each image
    def plot_depth_metrics(self):
        plt.figure()
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
        plt.figure()
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

    # Choose which mechanical model to use
    use_method = "naive"
    assert use_method in ["naive", "hertz", "MDR"]

    fig1 = plt.figure(1)
    sp1 = fig1.add_subplot(211)
    sp1.set_xlabel('Measured Sensor Deformation (d) [m]')
    sp1.set_ylabel('Force [N]')
    
    if use_method == "naive":
        # Set up stress / strain axes for naive method
        fig2 = plt.figure(2)
        sp2 = fig2.add_subplot(211)
        sp2.set_xlabel('dL / L')
        sp2.set_ylabel('F / A')

    elif use_method == "MDR":
        # Set up axes for MDR method
        fig2 = plt.figure(2)
        sp2 = fig2.add_subplot(211)
        sp2.set_xlabel('[Pa]^(-2/3)')
        sp2.set_ylabel('Depth [m]')

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

    # Unload data from folder
    data_folder = "./example_data/2023-12-16"
    data_files = os.listdir(data_folder)
    for i in range(len(data_files)):
        file_name = data_files[i]
        if os.path.splitext(file_name)[1] != '.avi':
            continue
        obj_name = os.path.splitext(file_name)[0].split('__')[0]

        if obj_name.count('foam') == 0: continue
        print('Object:', obj_name)

        # Load data and clip
        estimator = EstimateModulus(use_gripper_width=True)
        estimator.load_from_file(data_folder + "/" + os.path.splitext(file_name)[0], auto_clip=True)
        
        estimator.clip_to_press()
        assert len(estimator.depth_images()) == len(estimator.forces()) == len(estimator.gripper_widths())



        # fit_ellipse(estimator.contact_mask(estimator.depth_images()[-1]), plot_result=True)
        # fit_ellipse_float(estimator.depth_images()[-1], plot_result=True)


        ellipse_mask = []
        binary_array = []
        for i in range(len(estimator.depth_images())):

            # Normalize array
            float_array = estimator.depth_images()[i]
            float_array_normalized = (float_array - float_array.min()) / (float_array.max() - float_array.min())

            # Threshold into binary array based on range
            binary_array.append((255 * (float_array_normalized >= 0.5)).astype(np.uint8))

            max_ellipse = fit_ellipse_float(estimator.depth_images()[i])
            ellipse_image = np.zeros_like(estimator.depth_images()[i], dtype=np.uint8)
            cv2.ellipse(ellipse_image, max_ellipse, 255, -1)
            ellipse_mask.append(ellipse_image)

        plt.ion()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, title in zip(axes, ['Depth', 'Binary Thresholding', 'Ellipse Mask']):
            ax.set_title(title)
            ax.set_xlabel('Y')
            ax.set_ylabel('X')

        im1 = axes[0].imshow(estimator.depth_images()[0], cmap="winter")
        im2 = axes[1].imshow(binary_array[0], cmap=plt.cm.gray)
        im3 = axes[2].imshow(ellipse_mask[0], cmap=plt.cm.gray)

        plt.tight_layout()

        for i in range(len(estimator.depth_images())):
            im1.set_array(estimator.depth_images()[i])
            im2.set_array(binary_array[i])
            im3.set_array(ellipse_mask[i])
            
            plt.draw()
            plt.pause(0.5)

        plt.ioff()
        plt.show()


        if use_method == "naive":
            # Fit using naive estimator
            E_object = estimator.fit_modulus_naive(use_ellipse_fitting=True)

        elif use_method == "MDR":
            # Fit using our MDR estimator
            E_star = estimator.fit_modulus_MDR(use_ellipse_fitting=True)
            E_object, v_object = estimator.Estar_to_E(E_star)

        # print(f'Maximum depth of {obj_name}:', np.max(estimator.max_depths()))
        # print(f'Maximum force of {obj_name}:', np.max(estimator.forces()))
        # print(f'Strain range of {obj_name}:', min(estimator._x_data), 'to', max(estimator._x_data))
        # print(f'Stress range of {obj_name}:', min(estimator._y_data), 'to', max(estimator._y_data))
        # print(f'Contact radius range of {obj_name}:', min(estimator._a), 'to', max(estimator._a))
        # print(f'Depth range of {obj_name}:', min(estimator._d), 'to', max(estimator._d))
        # print(f'Mean radius of {obj_name}:', sum(estimator._R) / len(estimator._R))
        print(f'Estimated modulus of {obj_name}:', E_object)
        print('\n')

        # Plot
        plotting_color = random_shade_of_color(obj_to_color[obj_name])
        sp1.plot(estimator.max_depths(), estimator.forces(), ".", label=obj_name, markersize=8, color=plotting_color)
        sp2.plot(estimator._x_data, estimator._y_data, ".", label=obj_name, markersize=8, color=plotting_color)

        if use_method == "naive":
            # Plot naive fit
            sp2.plot(estimator._x_data, E_object*np.array(estimator._x_data), "-", label=obj_name, markersize=8, color=plotting_color)
            
        elif use_method == "MDR":
            # Plot MDR fit
            sp2.plot(estimator._x_data, np.array(estimator._x_data)*(E_star**(2/3)), "-", label=obj_name, markersize=8, color=plotting_color)

    fig1.legend()
    fig1.set_figwidth(10)
    fig1.set_figheight(10)
    fig2.legend()
    fig2.set_figwidth(10)
    fig2.set_figheight(10)
    plt.show()
    print('here')