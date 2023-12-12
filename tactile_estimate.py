import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt

from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce
from gripper_width import GripperWidth
from data_recorder import DataRecorder

data_recorder = DataRecorder()

from scipy.ndimage import convolve
from scipy.optimize import minimize

# Archived measurements from calipers on surface in x and y directions
WARPED_PX_TO_MM = (11, 11)
RAW_PX_TO_MM = (12.5, 11)

# Use sensor size and warped data to choose conversion
IMG_R, IMG_C = 300, 400
SENSOR_PAD_DIM_MM = (24, 33) # [mm]
PX_TO_MM = np.sqrt((IMG_R / SENSOR_PAD_DIM_MM[0])**2 + (IMG_C / SENSOR_PAD_DIM_MM[1])**2)
MM_TO_PX = 1/PX_TO_MM

class EstimateModulus():
    def __init__(self, depth_threshold=0.00005, assumed_poissons_ratio=0.45, edge_crop_margin=55, use_gripper_width=True):
        self.assumed_poisson_ratio = assumed_poissons_ratio # [\]
        self.depth_threshold = depth_threshold # [m]
        self.edge_crop_margin = edge_crop_margin # [pixels]
        self.use_gripper_width = use_gripper_width # Boolean of whether or not to include gripper width

        self.depth_images = []      # Depth images [in m]
        self.forces = []            # Measured contact forces
        self.gripper_widths = []    # Measured width of gripper

        self.F = []                 # Contact forces for fitting
        self.d = []                 # Contact depth for fitting
        self.a = []                 # Contact radius for fitting

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

    # Clear out the data values
    def _reset_data(self):
        self.depth_images = []
        self.forces = []
        self.gripper_widths = []
        self.F = []
        self.d = []
        self.a = []

    # Directly load measurement data
    def load(self, depth_images, forces, gripper_widths):
        self._reset_data()
        assert len(depth_images) == len(forces)
        self.depth_images = 0.001 * depth_images # Convert to [m]
        self.forces = forces
        self.gripper_widths = gripper_widths
        return

    # Load data from a file
    def load_from_file(self, path_to_file):
        self._reset_data()
        data_recorder = DataRecorder(use_gripper_width=self.use_gripper_width)
        data_recorder.load(path_to_file)
        data_recorder.auto_clip()

        # Extract the data we need
        if len(data_recorder.forces()) > len(data_recorder.depth_images()):
            data_recorder.contact_force.clip(0, len(data_recorder.depth_images()))
        # TODO: Fix this. We shouldn't need to clip
        assert len(data_recorder.depth_images()) == len(data_recorder.forces())
        self.depth_images = 0.001 * data_recorder.depth_images()
        self.forces = data_recorder.forces()
        if self.use_gripper_width: self.gripper_widths = data_recorder.widths()
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
    def crop_edges(self, depth, vertical_shift=20):
        assert depth.shape[0] > 2*self.edge_crop_margin and depth.shape[1] > 2*self.edge_crop_margin
        filtered_depth = depth.copy()
        filtered_depth[0:self.edge_crop_margin, :] = 0
        filtered_depth[:, 0:self.edge_crop_margin+vertical_shift] = 0
        filtered_depth[depth.shape[0]-self.edge_crop_margin:depth.shape[0], :] = 0
        filtered_depth[:, depth.shape[1]-self.edge_crop_margin+vertical_shift:depth.shape[1]] = 0
        return filtered_depth
    
    # Filter all depth images using masks and cropping
    def filter_depths(self, concave_mask=False, crop_edges=True):
        for i in range(self.depth_images.shape[0]):
            depth = self.depth_images[i,:,:]

            # Mask depth to consider contact area only
            contact_mask = self.contact_mask(depth)
            filtered_depth = depth * contact_mask

            if concave_mask: # Only consider convex points on surface
                filtered_depth = self.concave_mask(depth) * filtered_depth
            if crop_edges: # Remove edge regions which could be noisy
                filtered_depth = self.crop_edges(filtered_depth)
            self.depth_images[i,:,:] = filtered_depth

        return
    
    # Return maximum value in every depth image
    def max_depths(self, depth_images):
        return np.max(depth_images, axis=(1,2))
    
    # Return mean value in every depth image
    def mean_depths(self, depth_images):
        return np.mean(depth_images, axis=(1,2))

    # Fit linear equation with least squares
    def linear_coeff_fit(self, x, y):
        # Solve for best A given data and equation of form y = A*x
        return np.dot(x, y) / np.dot(x, x)
    
    # Cip a press sequence to only the loading sequence (positive force)
    def clip_to_press(self, pct_of_max=0.985):
        # Find maximum depth over press
        max_depths = self.max_depths(self.depth_images)
        i_start = np.argmax(max_depths >= self.depth_threshold)

        i_peak = np.argmax(max_depths)
        peak_depth = np.max(max_depths)
        for i in range(max_depths.shape[0]):
            if max_depths[i] > pct_of_max*peak_depth:
                i_peak = i
                break

        if i_start >= i_peak:
            warnings.warn("No press detected! Cannot clip.", Warning)
        else:
            # Clip from start to peak depth
            self.depth_images = self.depth_images[i_start:i_peak+1, :, :]
            self.forces = self.forces[i_start:i_peak+1]
            if self.use_gripper_width: 
                self.gripper_widths = self.gripper_widths[i_start:i_peak+1]
        return
    
    # Fit to continuous function and down sample to smooth measurements
    def smooth_gripper_widths(self, plot_smoothing=False, poly_order=5):
        smooth_widths = []
        indices = np.arange(len(self.gripper_widths))
        p = np.polyfit(indices, self.gripper_widths, poly_order)
        for i in indices.tolist():
            w = 0
            for k in range(len(p)):
                w += p[k] * i**(poly_order-k)
            smooth_widths.append(w)

        if plot_smoothing:
            # Plot to check how the smoothing of data looks
            plt.plot(indices, self.gripper_widths, 'r.')
            plt.plot(indices, smooth_widths, 'b-')
            plt.show()

        self.gripper_widths = np.array(smooth_widths)
        return
    
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

        # init_guess = [0.01, np.mean(X), np.mean(Y), np.mean(Z) - 0.01]

        # def sphere_res(params, x, y, z):
        #     r, cx, cy, cz = params
        #     distances = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        #     return np.sum((distances - r)**2)
        
        # def center_constraint(x):
        #     return -x[3] # c_z <= 0
        
        # def contact_constraint(x):
        #     return x[0] + x[3]  # r + c_z >= 0
        
        # def max_depth_constraint(x):
        #     return np.max(Z) - x[0] - x[3] # r + c_z <= max_depth
        
        # eps = 0.005
        # def xc1(x):
        #     return x[1] - np.mean(X) + eps
        # def xc2(x):
        #     return np.mean(X) - x[1] + eps
        # def yc1(x):
        #     return x[2] - np.mean(Y) + eps
        # def yc2(x):
        #     return np.mean(Y) - x[2] + eps

        # # Use optimization to minimize residuals
        # c = (   {"type": "ineq", "fun": center_constraint},
        #         {"type": "ineq", "fun": contact_constraint},
        #         {"type": "ineq", "fun": max_depth_constraint},
        #         {"type": "ineq", "fun": xc1},
        #         {"type": "ineq", "fun": xc2},
        #         {"type": "ineq", "fun": yc1},
        #         {"type": "ineq", "fun": yc2} )
        # result = minimize(sphere_res, init_guess, args=(X, Y, Z), constraints=c)

        # # Extract fitted parameters
        # r, cx, cy, cz = result.x

        # return [r, cx, cy, cz]
    
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

    # Compute radius of contact from sphere fit
    def estimate_contact_radius(self, sphere):
        return np.sqrt(max(sphere[0]**2 - sphere[3]**2, 0))

    # Compute depth of contact from sphere fit
    def estimate_contact_depth(self, sphere):
        return sphere[0] + sphere[3]
    
    # Return force, contact depth, and contact radius
    def _compute_contact_data(self):
        if len(self.F) > 0 and len(self.d) > 0 and len(self.a) > 0:
            assert len(self.F) == len(self.d) == len(self.a)
            return self.F, self.d, self.a
        
        # Fit to depth and radius for each frame
        F, d, a = [], [], []
        for i in range(self.depth_images.shape[0]):
            # sphere = [0, 0, 0, 0] # [r, cx, cy, cz]
            # if np.max(self.depth_images[i]) > 0:
            #     sphere = self.fit_depth_to_sphere(self.depth_images[i])

            # if 0 < self.estimate_contact_depth(sphere) and \
            #     0 < self.estimate_contact_radius(sphere) < self.gel_width/2:

            #     F.append(-self.forces[i])
            #     d.append(self.estimate_contact_depth(sphere))
            #     a.append(self.estimate_contact_radius(sphere))
            #     # self.plot_sphere_fit(self.depth_images[i], sphere)
            #     # if self.estimate_contact_radius(sphere) > self.gel_width/2:
            #     #     raise ValueError("Contact radius larger than sensor gel!")

            # Don't use sphere fitting at all...
            if np.max(self.depth_images[i]) >= self.depth_threshold:
                F.append(abs(self.forces[i]))
                # d.append(np.max(self.depth_images[i]))

                # Get average of top 10 values
                flattened_depth = self.depth_images[i].flatten()
                d.append(np.mean(np.sort(flattened_depth)[-10:]) )

                contact_area = (0.001 / PX_TO_MM)**2 * np.sum(self.depth_images[i] > 0.0000001)
                a.append(np.sqrt(contact_area / np.pi))

        # TODO: Fit d and a to strictly increasing / step function (e.g. sigmoid?)

        assert len(d) > 0, "Could not find any reasonable contact!"
        self.F = np.squeeze(np.array(F))
        self.d = np.squeeze(np.array(d))
        self.a = np.squeeze(np.array(a))
        return
    
    # Plot all data over indices
    def plot_grasp_data(self):
        plt.plot(abs(self.forces) / abs(self.forces).max(), label="Normalized Forces")
        plt.plot(self.gripper_widths / self.gripper_widths.max(), label="Normalized Gripper Width")
        plt.plot(self.max_depths(self.depth_images) / self.max_depths(self.depth_images).max(), label="Normalized Depth")
        plt.legend()
        plt.show()
        return
    
    # Plot force versus contact depth
    def plot_F_vs_d(self, plot_fit=True, plot_title=None):
        plt.figure()
        plt.xlabel('Depth [m]')
        plt.ylabel('Force [N]')
        plt.title(plot_title)

        self._compute_contact_data()
        plt.plot(self.d, self.F, 'r.', label="Raw measurements", markersize=10)
        
        if plot_fit:
            E_star = self.linear_coeff_fit((4/3)*self.a*self.d, self.F)
            F_fit = (4/3)*E_star*self.a*self.d
            plt.plot(self.d, F_fit, 'b-', label="Fit", markersize=10)

        plt.legend(); plt.show(block=False)
        return
    
    def Estar_to_E(self, E_star):
        # Compute compliance from E_star by assuming Poisson's ratio
        nu = self.assumed_poisson_ratio
        E = (1 - nu**2) / (1/E_star - (1 - self.nu_gel**2)/(self.E_gel))
        return E, nu
    
    def fit_modulus(self):
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

        for i in range(self.depth_images.shape[0]):
            F_i = abs(self.forces[i])
            contact_area = (0.001 / PX_TO_MM)**2 * np.sum(self.depth_images[i] > 0.0000001)
            a_i = np.sqrt(contact_area / np.pi)

            # Take mean of 5x5 neighborhood around maximum depth
            max_index = np.unravel_index(np.argmax(self.depth_images[i]), self.depth_images[i].shape)
            d_i = np.mean(self.depth_images[i][max_index[0]-2:max_index[0]+3, max_index[1]-2:max_index[1]+3])

            if F_i > 0 and a_i >= 0.003 and d_i > self.depth_threshold:
                p_0 = (1/np.pi) * (6*F_i/(R**2))**(1/3) # times E_star^2/3
                q_1D_0 = p_0 * np.pi * a_i / 2
                w_1D_0 = (1 - self.nu_gel**2) * q_1D_0 / self.E_gel
                F.append(F_i)
                d.append(d_i)
                a.append(a_i)
                x_data.append(w_1D_0)
                y_data.append(d_i)

        self.d = np.array(d)
        self.a = np.array(a)
        self.F = np.array(F)
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

        # Fit for E_star
        E_star = self.linear_coeff_fit(x_data, y_data)**(3/2)
        return E_star
    
    # Naively estimate modulus based on gripper width change and aggregate modulus
    def fit_modulus_naive(self, use_mean=True):
        assert self.use_gripper_width

        # Find initial length of first contact
        L0 = self.gripper_widths[0]
        d0 = 0
        
        contact_areas, a = [], []
        x_data, y_data, d = [], [], []
        for i in range(len(self.depth_images)):
            depth_i = self.depth_images[i]

            if use_mean:
                # d_i = depth_i.mean()
                d_i = depth_i[self.edge_crop_margin:depth_i.shape[0]-self.edge_crop_margin, \
                              self.edge_crop_margin+20:depth_i.shape[1]-self.edge_crop_margin+20].mean()
                contact_area_i = (0.001 / PX_TO_MM)**2 * (depth_i.shape[0] - 2*self.edge_crop_margin) * (depth_i.shape[1] - 2*self.edge_crop_margin)
            else:
                d_i = depth_i.max()
                contact_area_i = (0.001 / PX_TO_MM)**2 * np.sum(depth_i >= self.depth_threshold)

            a_i = np.sqrt(contact_area_i / np.pi)

            dL = -(self.gripper_widths[i] - L0 + 2*(d_i - d0))
            if contact_area_i >= 1e-5 and dL >= 0:
                x_data.append(dL/L0) # Strain
                y_data.append(abs(self.forces[i]) / contact_area_i) # Stress
                contact_areas.append(contact_area_i)
                a.append(a_i)
                d.append(d_i)

        # Save stuff
        self.x_data = x_data
        self.y_data = y_data
        self.contact_areas = contact_areas
        self.a = a
        self.d = d

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
        L0 = self.gripper_widths[0]
        d0 = 0

        x_data, y_data = [], []
        d, contact_areas, a = [], [], []
        for i in range(len(self.depth_images)):
            depth_i = self.depth_images[i]
            d_i = L0 - self.gripper_widths[i]
            
            # contact_area_i = (0.001 / PX_TO_MM)**2 * (depth_i.shape[0] - 2*self.edge_crop_margin) * (depth_i.shape[1] - 2*self.edge_crop_margin)
            contact_area_i = (0.001 / PX_TO_MM)**2 * np.sum(depth_i >= self.depth_threshold)

            a_i = np.sqrt(contact_area_i / np.pi)

            if contact_area_i >= 1e-5 and d_i > 0:
                x_data.append(2*d_i*a_i)
                y_data.append(self.forces[i])
                contact_areas.append(contact_area_i)
                d.append(d_i)
                a.append(a_i)

        self.x_data = x_data
        self.y_data = y_data
        self.contact_areas = contact_areas
        self.a = a
        self.d = d

        E_agg = self.linear_coeff_fit(x_data, y_data)
        E = (1/E_agg - 1/self.E_gel)**(-1)  

        return E

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

    # fig2 = plt.figure(2)
    # sp2 = fig2.add_subplot(211)
    # sp2.set_xlabel('Measured Sensor Deformation (d) [m]')
    # sp2.set_ylabel('Measured Contact Radius (a) [m]')
    
    # fig3 = plt.figure(3)
    # sp3 = fig3.add_subplot(211)
    # sp3.set_xlabel('d*a [m^2]')
    # sp3.set_ylabel('Force [N]')

    wedge_video    = GelsightWedgeVideo(config_csv="./config.csv") # Force-sensing finger
    contact_force  = ContactForce()
    data_recorder  = DataRecorder(wedge_video=wedge_video, contact_force=contact_force, use_gripper_width=True)

    objs = [
            "foam_brick_1", "foam_brick_2", "foam_brick_3", \
            "orange_ball_softest_1", "orange_ball_softest_2", "orange_ball_softest_3", \
            "green_ball_softer_1", "green_ball_softer_2", "green_ball_softer_3", \
            # "blue_ball_harder_1", "blue_ball_harder_2", "blue_ball_harder_3", \
            "purple_ball_hardest_1", "purple_ball_hardest_2", "purple_ball_hardest_3", \
            "golf_ball_1", "golf_ball_2", "golf_ball_3", \
        ]
    plotting_colors = [
        "#FF3131", "#C41E3A", "#800020", \
        "#FFAC1C", "#FF7F50", "#FFD700", \
        "#50C878", "#4CBB17", "#355E3B", \
        # "#89CFF0", "#0096FF", "#0000FF", \
        "#BF40BF", "#CF9FFF", "#5D3FD3", \
        "#89CFF0", "#0096FF", "#0000FF", \
    ]
    for i in range(len(objs)):
        obj_name = objs[i]

        # Load data and clip
        estimator = EstimateModulus(use_gripper_width=True)
        estimator.load_from_file("./example_data/2023-12-06/" + obj_name)
        estimator.filter_depths(concave_mask=False)
        estimator.clip_to_press()
        assert len(estimator.depth_images) == len(estimator.forces) == len(estimator.gripper_widths)

        estimator.smooth_gripper_widths()

        # # Fit using our MDR estimator
        # E_star = estimator.fit_modulus()
        # E_object, v_object = estimator.Estar_to_E(E_star)

        E_object = estimator.fit_modulus_hertz()

        print(f'Maximum depth of {obj_name}:', np.max(estimator.max_depths(estimator.depth_images)))
        print(f'Maximum force of {obj_name}:', np.max(estimator.forces))
        print(f'Strain range of {obj_name}:', min(estimator.x_data), 'to', max(estimator.x_data))
        print(f'Stress range of {obj_name}:', min(estimator.y_data), 'to', max(estimator.y_data))
        print(f'Contact radius range of {obj_name}:', min(estimator.a), 'to', max(estimator.a))
        print(f'Depth range of {obj_name}:', min(estimator.d), 'to', max(estimator.d))
        print(f'Estimated modulus of {obj_name}:', E_object)
        print('\n')

        # Plot
        sp1.plot(estimator.max_depths(estimator.depth_images), estimator.forces, ".", label=obj_name, markersize=8, color=plotting_colors[i])
        sp2.plot(estimator.x_data, estimator.y_data, ".", label=obj_name, markersize=8, color=plotting_colors[i])

        # sp1.plot(estimator.d, estimator.F, ".", label=obj_name, markersize=8, color=plotting_colors[i])
        # sp2.plot(estimator.d, estimator.a, ".", label=obj_name, markersize=8, color=plotting_colors[i])
        # sp3.plot(estimator.d*estimator.a, estimator.F, ".", label=obj_name, markersize=8, color=plotting_colors[i])

        # F_fit = []
        # R = 0.025
        # for j in range(len(estimator.d)):
        #     d_i = estimator.d[j]
        #     a_i = estimator.a[j]
        #     q_1D_0 = d_i * estimator.E_gel / (1 - estimator.nu_gel**2)
        #     p_0 = 2 * q_1D_0 / (np.pi * a_i)
        #     F_fit_i = (p_0 * np.pi)**3 * (R**2 / E_star**2) / 6
        #     F_fit.append(F_fit_i)
        # sp1.plot(estimator.d, F_fit, "-", label=(obj_name + " fit"), markersize=10, color=plotting_colors[i])

    fig1.legend()
    fig1.set_figwidth(10)
    fig1.set_figheight(10)
    fig2.legend()
    fig2.set_figwidth(10)
    fig2.set_figheight(10)
    # fig3.legend()
    # fig3.set_figwidth(10)
    # fig3.set_figheight(10)
    plt.show()