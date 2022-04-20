import logging
import os
from glob import glob

import natsort
import numpy as np
import pydicom
from numba import njit


@njit(cache=True)
def transform_to_hu(image: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """
    A function to transform a ct scan image into hounsfield units \n
    :param image: a numpy array containing the raw ct scan
    :param slope: the slope rescaling factor from the dicom file (0 if already in hounsfield units)
    :param intercept: the intercept value from the dicom file (depends on the machine)
    :return: a copy of the numpy array converted into hounsfield units
    """
    hu_image = image.copy()
    hu_image = hu_image * slope + intercept
    return hu_image


@njit(cache=True)
def window_image(image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    """
    A function to window the hounsfield units of the ct scan \n
    :param image: a numpy array containing the hounsfield ct scan
    :param window_center: hounsfield window center
    :param window_width: hounsfield window width
    :return: a windowed copy of 'image' parameter
    """
    # Get the min/max hounsfield units for the dicom image
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    windowed_img = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < img_min:
                windowed_img[i, j] = img_min
            elif image[i, j] > img_max:
                windowed_img[i, j] = img_max
            else:
                windowed_img[i, j] = image[i, j]

    return windowed_img


class DicomSeries:
    """
    A class representing a DicomSeries in a specified directory\n
    :param directory: the directory containing the dicom series
    :param pattern: the file pattern of the dicom series
    :param window_center: the center of the hounsfield units to view the images (default: 30)
    :param window_width: the width of the hounsfield units to view the images (default: 150)
    """

    def __init__(self, directory: str):
        self.directory = directory
        self.series_info = self._get_image_info()
        self.mrn = self.series_info['mrn']
        self.acc = self.series_info['accession']
        self.cut = self.series_info['cut']
        # self.hounsfield_array = self.read_dicom_series(pattern, window_center, window_width)

    def _get_image_info(self):
        """
        A method to retrieve the series information\n
        :return: a dictionary containing the information for the series
        """

        # establish the home directory for the series
        directory = self.directory
        # Create the empty dictionary
        series_info = {}
        # Read in an example dicom file from the directory
        first_file = os.listdir(directory)[0]
        filepath = f'{directory}/{first_file}'
        ds = pydicom.dcmread(filepath)

        default = 'NA'
        attributes = [
            'PatientID',
            'AccessionNumber',
            'SeriesDescription',
            'ImageOrientationPatient',
            'PatientSex',
            'PatientBirthDate',
            'PatientAge',
            'AcquisitionDate',
            'PixelSpacing',
            'SliceThickness',
            'RescaleSlope',
            'RescaleIntercept',
            'KVP',
            'Manufacturer',
            'ManufacturerModelName'
        ]

        attr_dict = {
            'PatientID': 'mrn',
            'AccessionNumber': 'accession',
            'SeriesDescription': 'cut',
            'ImageOrientationPatient': 'ct_direction',
            'PatientSex': 'sex',
            'PatientBirthDate': 'birthday',
            'PatientAge': 'age_at_scan',
            'AcquisitionDate': 'scan_date',
            'RescaleSlope': 'slope',
            'RescaleIntercept': 'intercept',
            'PixelSpacing': 'pixel_spacing',
            'SliceThickness': 'slice_thickness',
            'KVP': 'kvp',
            'Manufacturer': 'manufacturer',
            'ManufacturerModelName': 'manufacturer_model'
        }

        coordinates = {
            'COR': [1, 0, 0, 0, 0, -1],
            'SAG': [0, 1, 0, 0, 0, -1],
            'AX': [1, 0, 0, 0, 1, 0]
        }

        for attribute in attributes:
            column_name = attr_dict[attribute]
            if hasattr(ds, attribute) and getattr(ds, attribute) is not None:
                # Get the series description
                if attribute == 'SeriesDescription':
                    description = ds.SeriesDescription.replace('/', '_').replace(' ', '_').replace('-', '_').replace(
                        ',', '.')
                    series_info[column_name] = description
                # Determine if the scan is sagittal, coronal, or axial
                elif attribute == 'ImageOrientationPatient':
                    orientation = [np.int(x) for x in ds.ImageOrientationPatient]
                    for k, v in coordinates.items():
                        if orientation == v:
                            series_info[column_name] = k
                # Calculate birthday
                elif attribute == 'PatientBirthDate':
                    bday = ds.PatientBirthDate
                    d, m, y = bday[-2:], bday[-4:-2], bday[:4]
                    series_info[column_name] = '/'.join([d, m, y])

                elif attribute == 'PatientAge':
                    series_info[column_name] = int(ds.PatientAge[:-1])

                elif attribute == 'AcquisitionDate':
                    acq_date = ds.AcquisitionDate
                    series_info[column_name] = '/'.join([acq_date[-2:], acq_date[-4:-2], acq_date[:4]])
                # Get voxel dimensions
                elif attribute == 'PixelSpacing':
                    length, width = ds.PixelSpacing
                    length = length / 10
                    width = width / 10
                    series_info['length'] = length
                    series_info['width'] = width

                elif attribute == 'SliceThickness':
                    thickness = ds.SliceThickness
                    thickness = thickness / 10
                    series_info['thickness'] = thickness

                else:
                    series_info[column_name] = getattr(ds, attribute)

            else:
                series_info[column_name] = default
                continue

        # Get the voxel dimensions (in cm)

        # Get the hounsfield conversion factors
        # slope = ds.RescaleSlope
        # intercept = ds.RescaleIntercept
        #
        # # Create the dictionary
        #
        #
        # series_info['slope'] = slope
        # series_info['intercept'] = intercept
        #
        # # Jeff's information for covariates
        # series_info['kvp'] = ds.KVP
        # series_info['manufacturer'] = ds.Manufacturer
        # series_info['manufacturer_model'] = ds.ManufacturerModelName
        # print(series_info)
        scan_types = ['AX', 'SAG', 'COR']
        if series_info['ct_direction'] == default:
            for stype in scan_types:
                if stype in series_info['cut'].upper():
                    series_info['ct_direction'] = stype
                    break
        return series_info

    def read_dicom_series(self, pattern: str, win_center: int, win_width: int) -> np.ndarray:
        """
        A method to gather the raw ct scan in hounsfield units\n
        :param pattern: the file name pattern of the dicom files in the directory (default: '*.dcm')
        :param win_center: the hounsfield window center of choice (default: 30)
        :param win_width: the hounsfield window width of choice (default: 150)
        :return: a 3-Dimensional numpy array of the ct scan
        """

        # Get the directory of the files
        directory = self.directory

        # Pull the info needed for hounsfield transformation
        slope = self.series_info['slope']
        intercept = self.series_info['intercept']

        if not os.path.exists(directory) or not os.path.isdir(directory):
            raise ValueError(f'Given directory does not exist or is not a file: {directory}')

        logging.info(f'Reading Dicom files from {directory}...')

        # Get the list of dicom files in the directory
        dicoms = natsort.natsorted(glob(os.path.join(directory, pattern)))
        logging.info(f'- Dicom series contains {len(dicoms)} images.')

        # Create an empty list to hold dicom arrays
        ct_scans = {}

        logging.info(f'- Transforming all dicom files in directory {directory} to hounsfield units...')
        # Loop through all the dicom files
        for dicom in dicoms:
            # load in the data
            ds = pydicom.read_file(dicom)
            # Convert into numeric array
            image = ds.pixel_array
            # Get the slide number for the image (can be out of order)
            slide_n = np.int(ds.InstanceNumber)

            # Convert to hounsfield and window array
            image = transform_to_hu(image, slope=slope, intercept=intercept)
            image = window_image(image, window_center=win_center, window_width=win_width)
            # append the image to the stack
            ct_scans[slide_n] = image

        sorted_ct = []
        for i in sorted(ct_scans.keys()):
            slide = i
            sorted_ct.append(ct_scans[slide])

        # Stack into a 3d array
        full_img = np.stack(sorted_ct, axis=2)
        logging.info('- Dicom files have been combined into one volume.\n')
        return full_img
