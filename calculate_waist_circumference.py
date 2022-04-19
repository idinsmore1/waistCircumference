import argparse

import numpy as np
import pandas as pd
from numba import njit
from skimage import measure

from dicomseries import DicomSeries

# This is a sample Python script.
parser = argparse.ArgumentParser(description='Automatically Calculate Waist Circumference from CT Scan')
parser.add_argument('-i', '--input', help='Input DICOM Series Directory', required=True)
parser.add_argument('-o', '--output', help='Output CSV File Directory', required=True)
args = parser.parse_args()


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
    waist_pixels = measure.perimeter(body_array)
    waist_cm = waist_pixels * pixel_width
    return waist_cm


def main():
    dicom_series = DicomSeries(args.input)
    description = f'{dicom_series.mrn}_{dicom_series.acc}_{dicom_series.cut}'
    outfile = f'{args.output}/{description}.csv'
    ct_scan = dicom_series.read_dicom_series('*', -10, 300)
    waist_circumferences = []
    for image in range(ct_scan.shape[2])
        binary_im = binarize_scan(ct_scan[:, :, image])
        binary_im = fill_binary_gaps(binary_im)
        body_array = mark_body(binary_im)
        measurement = measure_circumference(body_array, dicom_series.series_info['width'])
        waist_circumferences.append(measurement)
    df = pd.DataFrame(waist_circumferences, columns=['waist_circumference'])
    df.to_csv(outfile)


if __name__ == '__main__':
    main()
