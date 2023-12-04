import cv2
import numpy as np
from PIL import Image, ImageOps
from ssim import SSIM
from skimage.metrics import structural_similarity as ssim
import numpy as np
from scipy.ndimage import zoom
from scipy.fftpack import fft2


def calculate_ssim(reference, enhanced):
    # Ensure the images are in the range [0, 255]
    reference = np.clip(reference, 0, 255).astype(np.uint8)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # Convert images to grayscale
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

    # Calculate SSIM for each channel
    ssim_r = ssim(reference[:, :, 0], enhanced[:, :, 0])
    ssim_g = ssim(reference[:, :, 1], enhanced[:, :, 1])
    ssim_b = ssim(reference[:, :, 2], enhanced[:, :, 2])

    # Average the SSIM values
    ssim_avg = (ssim_r + ssim_g + ssim_b) / 3.0

    return ssim_avg


def calculate_cwssim(reference, enhanced):
    # Ensure the images are in the range [0, 255]
    reference = np.clip(reference, 0, 255).astype(np.uint8)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # Convert images to grayscale
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

    reference = Image.fromarray(reference)
    enhanced = Image.fromarray(enhanced)

    # Calculate SSIM for each channel
    cwssim = SSIM(reference).cw_ssim_value(enhanced)

    return cwssim


def calculate_summer(im1, im2):
    outCT = 0

    for c in range(3):
        inp = np.double(im1[:, :, c]) / 255 - np.double(im2[:, :, c]) / 255
        for j in range(1, 5):
            inpR = zoom(inp, 1 / 2 ** j, order=1)
            out = np.mean(np.abs(np.log(1 + np.abs(fft2(np.abs(inpR))))))
            outCT += out

    for c in range(3):
        for j in range(3, 5):
            x1 = np.abs(fft2(zoom(np.double(im1[:, :, c]) / 255, 1 / 2 ** j, order=1)))
            x2 = np.abs(fft2(zoom(np.double(im2[:, :, c]) / 255, 1 / 2 ** j, order=1)))
            outCT += np.log(1 + np.mean(x1 / x2))

    outCT = 5 * np.power(1 / (outCT + 1), 1 / 3)

    return outCT


def PSNR(original, distorted, max_pixel=255.0):
    mse = np.mean((original - distorted) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value
