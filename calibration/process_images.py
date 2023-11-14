import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import circle_fit as cf

def pick_points_cv(img, circle_size=1, text="image"):
    img = img.copy()
    i = 0
    coords = []
    def click_event(event, x, y, flags, params):
        nonlocal i, coords
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append([x,y])
            cv2.circle(img, (x,y), circle_size, (0,255,0), -1)
            i += 1
            cv2.imshow(text, img)
    cv2.imshow(text, img)
    cv2.setMouseCallback(text, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    coords = np.array(coords).astype(np.float32)
    return img, coords

# Process calibration files into data
calibration_files = os.listdir('./data')
RGBXY = np.zeros((len(calibration_files), 5))
approx_GxGy = np.zeros((len(calibration_files), 2))
GxGy = np.zeros((len(calibration_files), 2))
for filename in os.listdir('./data'):

    # Read image
    img = cv2.imread("./data/" + filename)
    _,coords = pick_points_cv(img)
    xc,yc,r,_ = cf.least_squares_circle(coords)
    mask = np.zeros_like(img)
    cv2.circle(mask, (int(xc),int(yc)), int(r), (255, 0, 0) , -1)
    plt.imshow(mask)

    '''
    # Compute gradients and save to file
    for x in range():
        for y in range():
            dx = (diff_img[:, :, 1] - (diff_img[:, :, 0] + diff_img[:, :, 2]) * 0.5) # / 255.0
            dy = (diff_img[:, :, 0] - diff_img[:, :, 2]) # / 255.0
            approx_Gx = dx / (1 - dx ** 2) ** 0.5 / 128
            approx_Gy = dy / (1 - dy ** 2) ** 0.5 / 128

            denom = np.sqrt(r**2 - (x - xc)**2 - (y - yc)**2)
            G_x = -(x-xc) / denom
            G_y = -(y-yc) / denom

            RGBXY[i, :] = 
            approx_GxGy[i, :] = 
            GxGy[i, :] = 
    '''