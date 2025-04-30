import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from skimage.measure import regionprops
from skimage.draw import rectangle

# Define the 2D Gaussian model
def gaussian2D(coords, sigma_x, sigma_y, theta):
    x, y = coords[:, 0], coords[:, 1]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    a = (cos_theta ** 2) / (2 * sigma_x ** 2) + (sin_theta ** 2) / (2 * sigma_y ** 2)
    b = -(sin_theta * cos_theta) / (2 * sigma_x ** 2) + (sin_theta * cos_theta) / (2 * sigma_y ** 2)
    c = (sin_theta ** 2) / (2 * sigma_x ** 2) + (cos_theta ** 2) / (2 * sigma_y ** 2)
    return np.exp(-(a * x**2 + 2 * b * x * y + c * y**2))

def compute_A_beam(psf):
    # Crop the PSF
    psf = psf.squeeze().numpy(force=True)
    props = regionprops((psf > 0.5).astype(int))
    bounding_box = np.round(props[0].bbox).astype(int)
    roi = psf[
        bounding_box[0]:bounding_box[2],
        bounding_box[1]:bounding_box[3]
    ]

    # Center and grid
    y0, x0 = np.argwhere(roi == 1)[0]  # Find the center
    y, x = np.meshgrid(
        np.arange(roi.shape[1]) - x0,
        np.arange(roi.shape[0]) - y0
    )

    # Flattened data
    data = roi.flatten()
    coords = np.column_stack((x.flatten(), y.flatten()))


    # Initial guess: [sigma_x, sigma_y, theta]
    initial_guess = [
        roi.shape[1] / (2 * np.sqrt(2 * np.log(2))),
        roi.shape[0] / (2 * np.sqrt(2 * np.log(2))),
        0
    ]

    # Perform fitting
    params_fit, _ = curve_fit(
        lambda coords_flat, sigma_x, sigma_y, theta: gaussian2D(coords, sigma_x, sigma_y, theta).flatten(),
        coords, data, p0=initial_guess
    )

    # Extract fitted parameters
    sigma_x, sigma_y, theta = params_fit

    # Sigma to FWHM
    bmaj_pix = np.sqrt(8 * np.log(2)) * sigma_x
    bmin_pix = np.sqrt(8 * np.log(2)) * sigma_y

    # A beam value for normalisation
    A_beam = np.pi * bmaj_pix * bmin_pix / (4 * np.log(2))
    return A_beam