import logging
import os
from glob import glob
from datetime import datetime
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
    hu_image = image * slope + intercept
    return hu_image


@njit(cache=True)
def window_image(image: np.ndarray, window_center: int, window_width: int):
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

    new_image = np.clip(image, img_min, img_max)
    return new_image


class DicomSeries:
    """
    A class representing a series of dicom files in a specified directory\n
    :param directory: the directory containing the series
    :param filepattern: the file pattern of the series
    :param window_center: the center of the hounsfield units to view the images (default: 30)
    :param window_width: the width of the hounsfield units to view the images (default: 150)
    :param read_images: a boolean indicating whether to read the images or not (default: True)
    """
    orientations = {
        'COR': [1, 0, 0, 0, 0, -1],
        'SAG': [0, 1, 0, 0, 0, -1],
        'AX': [1, 0, 0, 0, 1, 0]
    }
    important_attributes = {
        'PatientID': 'mrn',
        'AccessionNumber': 'accession',
        'SeriesDescription': 'cut',
        'ImageOrientationPatient': 'ct_direction',
        'PatientSex': 'sex',
        'PatientBirthDate': 'birthday',
        'AcquisitionDate': 'scan_date',
        'PatientAge': 'age_at_scan',
        'PixelSpacing': 'pixel_spacing',
        'SliceThickness': 'slice_thickness_mm',
        'Manufacturer': 'manufacturer',
        'ManufacturerModelName': 'manufacturer_model',
        'KVP': 'kvp',
        'ContrastBolusAgent': 'contrast_bolus_agent',
        'ContrastBolusRoute': 'contrast_bolus_route',
        'MultienergyCTAcquisitionSequence': 'multienergy_ct'
    }

    def __init__(self, directory, filepattern='*.dcm', window_center=30, window_width=150, read_images=True):

        if not os.path.exists(directory) or not os.path.isdir(directory):
            raise ValueError(f'Given directory does not exist or is not a file: {directory}')

        self.directory = os.path.abspath(directory)
        self.header = pydicom.dcmread(glob(os.path.join(directory, filepattern))[0])
        if read_images:
            self.pixel_array = self._read_dicom_series(directory, filepattern, self.header.RescaleSlope,
                                                       self.header.RescaleIntercept, window_center, window_width)
        self.series_info = self._get_image_info(self.header)

    @staticmethod
    def _read_dicom_series(directory, filepattern, slope, intercept, window_center, window_width):
        """
        A function to read a series of dicom files into one numpy array \n
        :param directory: directory containing the files
        :param files: list of all files in the directory
        :param slope: the slope rescaling factor from the dicom file (0 if already in hounsfield units)
        :param intercept: the intercept value from the dicom file (depends on the machine)
        :param window_center: hounsfield window center
        :param window_width: hounsfield window width
        :return: a numpy array containing the dicom series
        """
        logging.info(f'Reading Dicom files from {directory}...')
        files = natsort.natsorted(glob(os.path.join(directory, filepattern)))
        logging.info(f'- Number of files: {len(files)}')

        # create an empty dictionary to store image and image number
        ct_scans = {}
        # read all the files
        logging.info(f'- Reading files...')
        for i, file in enumerate(files):
            ds = pydicom.dcmread(file)
            # get the image number
            if hasattr(ds, 'InstanceNumber'):
                image_number = int(ds.InstanceNumber)
            else:
                image_number = i
            image = ds.pixel_array
            # store the image and number in the dictionary
            ct_scans[image_number] = image

        # sort the images by image number
        logging.info(f'- Sorting images by image number...')
        sorted_ct = [ct_scans[key] for key in sorted(ct_scans.keys())]
        # stack the images into one array
        combined_image = np.stack(sorted_ct, axis=0)
        # Convert to hounsfield units
        logging.info(f'- Converting to hounsfield units...')
        hu_image = transform_to_hu(combined_image, slope, intercept)
        # Window the hounsfield units
        logging.info(f'- Windowing hounsfield units...')
        windowed_image = window_image(hu_image, window_center, window_width)
        # return the windowed image
        logging.info(f'- Done!\n')
        return windowed_image

    def _get_image_info(self, header):
        """
        A function to get the important information from the dicom header \n
        :param header: the dicom header object
        :return: a dictionary containing the important information
        """
        series_info = {}
        # Loop through all the important attributes
        for tag, column in self.important_attributes.items():
            try:
                value = getattr(header, tag)
                if tag == "SeriesDescription":
                    value = value.replace('/', '_').replace(' ', '_').replace('-', '_').replace(
                        ',', '.')
                elif tag == "ImageOrientationPatient":
                    orientation = np.round(getattr(header, tag))
                    for key, direction in self.orientations.items():
                        if np.array_equal(orientation, direction):
                            value = key
                            break
                elif tag == 'PatientBirthDate':
                    value = datetime.strptime(value, '%Y%m%d').date()
                    value = value.strftime('%Y-%m-%d')
                elif tag == 'AcquisitionDate':
                    value = datetime.strptime(value, '%Y%m%d').date()
                    value = value.strftime('%Y-%m-%d')
                elif tag == 'PatientAge':
                    value = int(value[:-1])
                elif tag == 'PixelSpacing':
                    # Value in this case is a list
                    length, width = value
                    series_info['length_mm'] = length
                    series_info['width_mm'] = width
                    continue
                elif tag == 'PatientSex':
                    if value == 'M':
                        value = 1
                    elif value == 'F':
                        value = 0
                elif tag == 'ContrastBolusAgent' or tag == 'ContrastBolusRoute':
                    value = True
                elif tag == 'MultienergyCTAcquisitionSequence':
                    value = True

                series_info[column] = value

            except AttributeError:
                logging.info(f'{tag} not found in dicom header')
                series_info[column] = None

        return series_info
