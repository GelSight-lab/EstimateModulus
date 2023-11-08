import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from scipy.optimize import minimize

# Measured from calipers on surface in x and y directions
WARPED_PX_TO_MM = (11, 11)
RAW_PX_TO_MM = (12.5, 11)
PX_TO_MM = np.sqrt(WARPED_PX_TO_MM[0]**2 + WARPED_PX_TO_MM[1]**2)

class TactileMaterialEstimate():
    def __init__(self, depth_threshold=0.05, assumed_poissons_ratio=0.45):
        self.assumed_poisson_ratio = assumed_poissons_ratio # [\]
        self.depth_threshold = depth_threshold # [mm]

        # Gel material: Silicone XP-565
        # Datasheet:
        #     "https://static1.squarespace.com/static/5b1ecdd4372b96e84200cf1d/t/5b608e1a758d46a989c0bf93/1533054492488/XP-565+%281%29.pdf"
        # Estimated from Shore A hardness:
        #     "https://www.dow.com/content/dam/dcc/documents/en-us/tech-art/11/11-37/11-3716-01-durometer-hardness-for-silicones.pdf"
        self.shore_A_00_hardness = 58 # From durometer measurement
        self.gel_compliance = 5.28e+6 # N/m^2
        self.gel_poisson_ratio = 0.485 # [\]

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

    # Fit linear equation with least squares
    def linear_coeff_fit(self, x, y):
        # Solve for best A given data and equation of form y = A*x
        return np.dot(x, y) / np.dot(x, x)
    
    # Crop a press sequence to only the loading sequence (positive force)
    def crop_press(self, depth, force):
        max_depth = np.zeros((depth.shape[0], 1))
        for i in range(depth.shape[0]):
            max_depth[i] = depth[i].max()

        i_start = np.argmin(max_depth >= self.depth_threshold)
        i_peak = np.argmax(max_depth)

        if i_start >= i_peak:
            warnings.warn("No press detected! Cannot crop.", Warning)
            cropped_depth, cropped_force = depth, force
        else:
            cropped_depth = depth[i_start:i_peak, :, :]
            cropped_force = force[i_start:i_peak]

        return cropped_depth, cropped_force
    
    # Convert depth image to 3D data
    def depth_to_XYZ(self, depth, concave_mask=True, crop_edges=True):
        # Mask depth to consider contact area only
        contact_mask = self.contact_mask(depth)
        filtered_depth = depth * contact_mask

        if concave_mask: # Only consider convex points on surface
            filtered_depth = self.concave_mask(depth) * filtered_depth
        if crop_edges: # Remove edge regions which could be noisy
            filtered_depth = self.crop_edges(filtered_depth)

        # Extract data
        X, Y, Z = [], [], []
        for i in range(filtered_depth.shape[0]):
            for j in range(filtered_depth.shape[1]):
                if abs(filtered_depth[i][j]) >= 0.0001:
                    X.append(0.001 * i / PX_TO_MM) # Convert to meters
                    Y.append(0.001 * j / PX_TO_MM)
                    Z.append(0.001 * filtered_depth[i][j])
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
        C, _, _, _ = np.linalg.lstsq(A, f)

        # Solve for the radius
        radius = np.sqrt((C[0]*C[0]) + (C[1]*C[1]) + (C[2]*C[2]) + C[3])

        return [radius, C[0], C[1], C[2]] # [ radius, center_x, center_y, center_z ]
    
    # Check sphere fit by plotting data and fit shape
    def plot_sphere_fit(self, depth, sphere):
        # Extract 3D data
        X, Y, Z = self.depth_to_XYZ(depth)

        # Create discrete graph of sphere mesh
        r, x0, y0, z0 = sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.cos(-z0/r):10j]
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
        return np.sqrt(sphere[0]**2 - sphere[3]**2)

    # Compute depth of contact from sphere fit
    def estimate_contact_depth(self, sphere):
        return sphere[0] + sphere[3]
    
    # Return force, contact depth, and contact radius
    def _get_contact_data(self, depth_images, forces):
        # Fit to depth and radius for each frame
        F, d, a = [], [], []
        for i in range(depth_images.shape[0]):
            sphere = self.fit_depth_to_sphere(depth_images[i])
            if self.estimate_contact_depth(sphere) > 0:
                F.append(forces[i])
                d.append(self.estimate_contact_depth(sphere))
                a.append(self.estimate_contact_radius(sphere))
        return F, d, a
    
    # Plot force versus contact depth
    def plot_F_vs_d(self, depth_images, forces):
        F, d, _ = self._get_contact_data(depth_images, forces)
        plt.plot(d, F, 'r.', label="Raw measurements", markersize=10)
        plt.xlabel('Depth [m]')
        plt.ylabel('Force [N]')
        plt.legend()
        plt.show()
        return

    # Use measured force and depth to estimate aggregate compliance E_star
    def fit_compliance(self, depth_images, forces):
        # Using Hertzian contact mechanics...
        #   dF/dd = 2*E_star*a
        # Following model from (2.3.2) in "Handbook of Contact Mechanics" by V.L. Popov

        # Fit to depth and radius for each frame
        F, d, a = self._get_contact_data(depth_images, forces)

        # Least squares regression for E_star
        dd = np.squeeze(np.diff(np.array(d), axis=0))
        dFdd = np.diff(np.array(F), axis=0) / np.clip(dd, 0.00001, 1)
        a = np.squeeze(np.array(a[:-1]))
        E_star = self.linear_coeff_fit(2*a, dFdd)

        # Compute compliance from E_star by assuming Poisson's ratio
        poisson_ratio = self.assumed_poisson_ratio
        E = (1 - poisson_ratio**2) / (1/E_star - (1 - self.gel_poisson_ratio**2)/(self.gel_compliance))
        return E, poisson_ratio

# RUN PRELIMINARY TEST TO MAKE SURE A SPHERE FOLLOWS ROUGHLY F ~ d^(2/3)
    # IF NOT, we are in large def / not purely elastic

if __name__ == "__main__":

    from data_recorder import DataRecorder
    data_recorder = DataRecorder()

    objs = ["foam_brick", "large_soft_sphere", "golf_ball", "small_rigid_sphere", "lego"]
    for obj_name in objs:
        # Load data and clip
        data_recorder.load("./example_data/" + obj_name)
        data_recorder.auto_clip()
        # data_recorder.wedge_video.watch()

        # Extract dynamic data
        depth_images = data_recorder.depth_images()
        press_force = data_recorder.forces()

        # Fit using our Hertzian estimator
        estimator = TactileMaterialEstimate()
        depth_images, press_force = estimator.crop_press(depth_images, press_force)
        E_finger, v_finger = estimator.fit_compliance(depth_images, press_force)
        print(f'\nEstimated modulus of {obj_name}:', E_finger, '\n')

    # TODO: TRY STOCHASTIC APPROACH???