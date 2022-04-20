import numpy as np
from numba import njit
from skimage import measure


@njit(parallel=True)
def binarize_scan(pixel_array: np.ndarray) -> np.ndarray:
    """
    Binarize a CT scan between air and body
    :param pixel_array: dicom pixel array
    :return: binary image as numpy array
    """
    binary_im = np.zeros(pixel_array.shape)
    binary_im[:450, :] = pixel_array[:450, :] > np.min(pixel_array) + 50
    binary_im = binary_im.astype(np.uint8)
    return binary_im


def fill_binary_gaps(binary_im: np.ndarray) -> np.ndarray:
    """
    Fill gaps in binary image
    :param binary_im: output from binarize_scan
    :return: filled binary image as numpy array
    """
    return np.maximum.accumulate(binary_im, 1) & \
           np.maximum.accumulate(binary_im[:, ::-1], 1)[:, ::-1] & \
           np.maximum.accumulate(binary_im[::-1, :], 0)[::-1, :] & \
           np.maximum.accumulate(binary_im, 0)


def mark_body(binary_im: np.ndarray) -> np.ndarray:
    """
    Remove all but the largest blob (the body) from a binary image
    :param binary_im: output from fill_binary_gaps
    :return: body only as a binary numpy array
    """
    body = measure.label(binary_im)
    body[body != 1] = 0
    return body


def measure_circumference(body_array, pixel_width):
    """
    Measure the circumference of the body in centimeters
    :param body_array: the output from mark_body
    :param pixel_width: the width of a pixel in centimeters
    :return: the circumference of the body in centimeters
    """
    waist_pixels = measure.perimeter(body_array)
    waist_cm = np.round(waist_pixels * pixel_width, 2)
    return waist_cm
