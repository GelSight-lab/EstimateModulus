import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt

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
    def __init__(self, depth_threshold=0.5, assumed_poissons_ratio=0.45):
        self.assumed_poisson_ratio = assumed_poissons_ratio # [\]
        self.depth_threshold = depth_threshold # [mm]

        self.depth_images = []      # Depth images [in mm]
        self.forces = []            # Measured contact forces
        self.F = []                 # Contact forces for fitting
        self.d = []                 # Contact depth for fitting
        self.dFdd = []              # Discrete derivative of force with respect to contact depth
        self.a = []                 # Contact radius for fitting

        # Gel material: Silicone XP-565
        # Datasheet:
        #     "https://static1.squarespace.com/static/5b1ecdd4372b96e84200cf1d/t/5b608e1a758d46a989c0bf93/1533054492488/XP-565+%281%29.pdf"
        # Estimated from Shore A hardness:
        #     "https://www.dow.com/content/dam/dcc/documents/en-us/tech-art/11/11-37/11-3716-01-durometer-hardness-for-silicones.pdf"
        self.shore_A_00_hardness = 58 # From durometer measurement
        self.gel_compliance = 5.28e+6 # N/m^2
        self.gel_poisson_ratio = 0.485 # [\]
        self.gel_width = 0.035 # [m]
        self.gel_depth = 0.0055 # [m] (slightly adjusted because of depth inaccuracies)

    # Clear out the data values
    def _reset_data(self):
        self.depth_images = []
        self.forces = []
        self.F = []
        self.d = []
        self.dFdd = []
        self.a = []

    # Directly load measurement data
    def load(self, depth_images, forces):
        self._reset_data()
        assert len(depth_images) == len(forces)
        self.depth_images = depth_images
        self.forces = forces
        return

    # Load data from a file
    def load_from_file(self, path_to_file):
        self._reset_data()
        data_recorder = DataRecorder()
        data_recorder.load(path_to_file)
        data_recorder.auto_clip()

        # Extract the data we need
        assert len(data_recorder.depth_images()) == len(data_recorder.forces())
        self.depth_images = data_recorder.depth_images()
        self.forces = data_recorder.forces()
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
    def crop_edges(self, depth, margin=55):
        assert depth.shape[0] > 2*margin and depth.shape[1] > 2*margin
        filtered_depth = depth.copy()
        filtered_depth[0:margin, :] = 0
        filtered_depth[:, 0:margin] = 0
        filtered_depth[depth.shape[0]-margin:depth.shape[0], :] = 0
        filtered_depth[:, depth.shape[1]-margin:depth.shape[1]] = 0
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
    def clip_press(self):
        # Find maximum depth over press
        max_depths = self.max_depths(self.depth_images)
        i_start = np.argmin(max_depths >= self.depth_threshold)
        i_peak = np.argmax(max_depths)

        if i_start >= i_peak:
            warnings.warn("No press detected! Cannot clip.", Warning)
        else:
            # Clip from start to peak depth
            self.depth_images = self.depth_images[i_start:i_peak, :, :]
            self.forces = self.forces[i_start:i_peak]
        return
    
    # Convert depth image to 3D data
    def depth_to_XYZ(self, depth):
        # Extract data
        X, Y, Z = [], [], []
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                if abs(depth[i][j]) >= 0.0001:
                    X.append(0.001 * i / PX_TO_MM) # Convert to meters
                    Y.append(0.001 * j / PX_TO_MM)
                    Z.append(0.001 * depth[i][j])
        data = np.vstack((np.array(X), np.array(Y), np.array(Z)))

        # Remove outliers via ~3-sigma rule
        X, Y, Z = [], [], []
        centroid = np.mean(data, axis=1)
        sigma = np.std(data, axis=1)
        for i in range(data.shape[1]):
            point = data[:, i]
            if (np.abs(centroid - point) <= 2.5*sigma).all():
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

        # # Construct data matrix
        # A = np.zeros((len(X),4))
        # A[:,0] = X*2
        # A[:,1] = Y*2
        # A[:,2] = Z*2
        # A[:,3] = 1

        # # Assemble the f matrix
        # f = np.zeros((len(X),1))
        # f[:,0] = (X*X) + (Y*Y) + (Z*Z)

        # # Solve least squares for sphere
        # C, res, rank, s = np.linalg.lstsq(A, f, rcond=None)

        # # Solve for the radius
        # radius = np.sqrt((C[0]*C[0]) + (C[1]*C[1]) + (C[2]*C[2]) + C[3])

        # return [radius, C[0], C[1], C[2]] # [ radius, center_x, center_y, center_z ]

        init_guess = [0.01, np.mean(X), np.mean(Y), np.mean(Z) - 0.01]

        def sphere_res(params, x, y, z):
            r, cx, cy, cz = params
            distances = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            return np.sum((distances - r)**2)
        
        def center_constraint(x):
            return -x[3] # c_z <= 0
        
        def contact_constraint(x):
            return x[0] + x[3]  # r + c_z >= 0
        
        def max_depth_constraint(x):
            return np.max(Z) - x[0] - x[3] # r + c_z <= max_depth
        
        eps = 0.005
        def xc1(x):
            return x[1] - np.mean(X) + eps
        def xc2(x):
            return np.mean(X) - x[1] + eps
        def yc1(x):
            return x[2] - np.mean(Y) + eps
        def yc2(x):
            return np.mean(Y) - x[2] + eps

        # Use optimization to minimize residuals
        c = (   {"type": "ineq", "fun": center_constraint},
                {"type": "ineq", "fun": contact_constraint},
                {"type": "ineq", "fun": max_depth_constraint},
                {"type": "ineq", "fun": xc1},
                {"type": "ineq", "fun": xc2},
                {"type": "ineq", "fun": yc1},
                {"type": "ineq", "fun": yc2} )
        result = minimize(sphere_res, init_guess, args=(X, Y, Z), constraints=c)

        # Extract fitted parameters
        r, cx, cy, cz = result.x

        return [r, cx, cy, cz]
    
    # Check sphere fit by plotting data and fit shape
    def plot_sphere_fit(self, depth, sphere):
        # Extract 3D data
        X, Y, Z = self.depth_to_XYZ(depth)

        # Create discrete graph of sphere mesh
        r, x0, y0, z0 = sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.cos(-z0/r):10j] # Plot top half of sphere
        # u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:20j] # Plot full sphere
        sphere_x = x0 + np.cos(u)*np.sin(v)*r
        sphere_y = y0 + np.sin(u)*np.sin(v)*r
        sphere_z = z0 + np.cos(v)*r

        # Plot sphere in 3D
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        axes.scatter(X, Y, Z, zdir='z', s=20, c='b', rasterized=True)
        axes.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r")
        axes.set_xlabel('$X$ [m]',fontsize=16)
        axes.set_ylabel('\n$Y$ [m]',fontsize=16)
        axes.set_zlabel('\n$Z$ [m]',fontsize=16)
        plt.show()

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
            sphere = [0, 0, 0, 0] # [r, cx, cy, cz]
            if np.max(self.depth_images[i]) > 0:
                sphere = self.fit_depth_to_sphere(self.depth_images[i])

            if 0 < self.estimate_contact_depth(sphere) and \
                0 < self.estimate_contact_radius(sphere) < self.gel_width/2:

                F.append(-self.forces[i])
                d.append(self.estimate_contact_depth(sphere))
                a.append(self.estimate_contact_radius(sphere))
                # self.plot_sphere_fit(self.depth_images[i], sphere)
                # if self.estimate_contact_radius(sphere) > self.gel_width/2:
                #     raise ValueError("Contact radius larger than sensor gel!")

        assert len(d) > 0, "Could not find any reasonable contact!"
        self.F = np.squeeze(np.array(F))
        self.d = np.squeeze(np.array(d))
        self.a = np.squeeze(np.array(a))
        return
    
    # Return force, contact depth, and contact radius
    def _compute_dFdd(self):
        # Fit to depth and radius for each frame
        if len(self.dFdd) > 0:
            assert len(self.F) == len(self.d) == len(self.a) == len(self.dFdd) + 1
            return self.dFdd
        dd = np.squeeze(np.diff(np.array(self.d), axis=0))
        self.dFdd = np.diff(np.array(self.F), axis=0) / np.clip(dd, 0.00001, 1)
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
            # self._compute_dFdd()
            # E_star = self.linear_coeff_fit(2*self.a[:-1], self.dFdd)
            E_star = self.linear_coeff_fit((4/3)*self.a*self.d, self.F)

            F_fit = np.zeros_like(self.F)
            F_fit[0] = self.F[0]
            for i in range(1, F_fit.shape[0]):
                # F_fit[i] = 2*self.a[i-1]*E_star*(self.d[i] - self.d[i-1]) + F_fit[i-1]
                F_fit[i] = (4/3)*self.a[i]*E_star*(self.d[i])
            plt.plot(self.d, F_fit, 'b-', label="Fit", markersize=10)

        plt.legend(); plt.show(block=False)
        return

    # Use measured force and depth to estimate aggregate compliance E_star
    def fit_compliance(self):
        # Using Hertzian contact mechanics...
        #   dF/dd = 2*E_star*a
        # Following model from (2.3.2) in "Handbook of Contact Mechanics" by V.L. Popov

        # Filter depth images to mask and crop
        self.filter_depths(concave_mask=False)

        # Fit to depth and radius for each frame
        self._compute_contact_data()

        # # Compute derivatives for fitting
        # self._compute_dFdd()

        # # Least squares regression for E_star
        # E_star = self.linear_coeff_fit(2*self.a[:-1], self.dFdd)

        E_star = self.linear_coeff_fit((4/3)*self.a*self.d, self.F)

        # Compute compliance from E_star by assuming Poisson's ratio
        poisson_ratio = self.assumed_poisson_ratio
        E = (1 - poisson_ratio**2) / (1/E_star - (1 - self.gel_poisson_ratio**2)/(self.gel_compliance))
        return E, poisson_ratio
    
    # Alternative method that tries to compute modulus pixel by pixel
    def fit_compliance_stochastic(self):
        target_data = []
        for i in range(1, self.depth_images.shape[0]):
            depth = self.depth_images[i]

            # Apply threshold mask
            contact_mask = self.contact_mask(depth)
            filtered_depth = 0.001 * depth * contact_mask

            # Remove edge regions which could be noisy
            filtered_depth = self.crop_edges(filtered_depth)

            # Collect data
            if np.max(filtered_depth) > 0:
                for r in range(depth.shape[0]):
                    for c in range(depth.shape[1]):
                        if depth[r][c] > 0:
                            # # Fit using derivatives
                            # dF = -(self.forces[i] - self.forces[i-1])
                            # du = self.depth_images[i][r][c] - self.depth_images[i-1][r][c]
                            # if dF/du > 0: target_data.append(dF / du)

                            # Fit using direct values
                            F = -self.forces[i]
                            u = depth[r][c]
                            if F/u > 0: target_data.append(F / u)

        # Use all data to fit function
        target_data = np.array(target_data)
        x_data = (0.001 * PX_TO_MM)**2 / self.gel_depth * np.ones_like(target_data)
        E_agg = self.linear_coeff_fit(x_data, target_data)
        
        # Calculate modulus of unknown object
        E = (1/E_agg - 1/self.gel_compliance)**(-1)

        return E
    

if __name__ == "__main__":

    ##################################################
    # GET ESTIMATED MODULUS (E) FOR SET OF TEST DATA #
    ##################################################

    objs = ["foam_brick_1", "foam_brick_2", "foam_brick_3", \
            "foam_earth_1", "foam_earth_2", "foam_earth_3", \
            "orange_ball_1", "orange_ball_2", "orange_ball_3", \
            "small_rigid_sphere_1", "small_rigid_sphere_2", "small_rigid_sphere_3", \
            "lego_1", "lego_2", "lego_3", \
        ]
    for obj_name in objs:
        
        # Load data and clip
        estimator = EstimateModulus()
        estimator.load_from_file("./example_data/2023-11-11/" + obj_name)
        assert len(estimator.depth_images) == len(estimator.forces)

        # Fit using our Hertzian estimator
        estimator.clip_press()
        E_finger, v_finger = estimator.fit_compliance()
        # E_finger = estimator.fit_compliance_stochastic()
        print(f'\nEstimated modulus of {obj_name}:', E_finger, '\n')

        # # Plot
        # estimator.plot_F_vs_d(plot_title=obj_name)

    """
    #####################################################
    # PLOT RAW DATA TO INVESTIGATE NOISE / CORRELATIONS #
    #####################################################

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)
    fig4 = plt.figure(4)
    fig5 = plt.figure(5)
    sp1 = fig1.add_subplot(211)
    sp2 = fig2.add_subplot(211)
    sp3 = fig3.add_subplot(211)
    sp4 = fig4.add_subplot(211)
    sp5 = fig5.add_subplot(211)
    sp1.set_title('Forces')
    sp2.set_title('Avg. Depths')
    sp3.set_title('Max Depths')
    sp4.set_title('Avg. Cropped Depths')
    sp5.set_title('Max Cropped Depths')

    objs = ["foam_brick_1", "foam_brick_2", "foam_brick_3", \
            "foam_earth_1", "foam_earth_2", "foam_earth_3", \
            "orange_ball_1", "orange_ball_2", "orange_ball_3", \
            "small_rigid_sphere_1", "small_rigid_sphere_2", "small_rigid_sphere_3", \
            "lego_1", "lego_2", "lego_3", \
        ]
    for obj_name in objs:
        
        # Load data and clip
        estimator = EstimateModulus()
        estimator.load_from_file("./example_data/2023-11-11/" + obj_name)
        assert len(estimator.depth_images) == len(estimator.forces)

        estimator.clip_press()
        x = range(estimator.depth_images.shape[0])
        sp1.plot(x, estimator.forces, label=obj_name)
        sp2.plot(x, estimator.mean_depths(estimator.depth_images), label=obj_name)
        sp3.plot(x, estimator.max_depths(estimator.depth_images), label=obj_name)

        estimator.filter_depths(concave_mask=False)
        sp4.plot(x, estimator.mean_depths(estimator.depth_images), label=obj_name)
        sp5.plot(x, estimator.max_depths(estimator.depth_images), label=obj_name)

    fig1.legend()
    fig2.legend()
    fig3.legend()
    fig4.legend()
    fig5.legend()
    plt.show()
    """