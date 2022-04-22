import numpy as np
from numba import njit
from skimage.measure import label, perimeter, regionprops


@njit()
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
    return np.maximum.accumulate(binary_im, 1) & np.maximum.accumulate(binary_im[:, ::-1], 1)[:,
                                                 ::-1] & np.maximum.accumulate(binary_im[::-1, :], 0)[::-1,
                                                         :] & np.maximum.accumulate(binary_im, 0)


def mark_body(binary_im: np.ndarray) -> np.ndarray:
    """
    Remove all but the largest blob (the body) from a binary image
    :param binary_im: output from fill_binary_gaps
    :return: body only as a binary numpy array
    """
    body = label(binary_im)
    body[body != 1] = 0
    return body


def measure_circumference(body_array, pixel_width):
    """
    Measure the circumference of the body in centimeters
    :param body_array: the output from mark_body
    :param pixel_width: the width of a pixel in centimeters
    :return: the circumference of the body in centimeters
    """
    waist_pixels = perimeter(body_array)
    waist_cm = np.round(waist_pixels * pixel_width, 2)
    return waist_cm


def detect_number_of_bones(pixel_array, upper_bound=175, area_threshold=100):
    """
    Detect the number of bones in the image
    :param pixel_array: the dicom pixel array
    :param upper_bound: the upper bound of the range of bone HU values
    :return: the number of bones in the image
    """
    bone_image = pixel_array >= upper_bound
    # Remove scan interference
    bone_image[400:, :] = 0
    # Label the number of bones
    labels = label(bone_image)
    regions = regionprops(labels)
    # remove small regions that are not important
    bones = [prop.label for prop in regions if prop.area > area_threshold]
    return len(bones)


def get_waist_range(waist_df):
    """
    Get the index of the waist in the dataframe
    :param waist_df: the dataframe with the waist measurements
    :return: the index of the waist in the dataframe
    """
    incr_ranges = {}
    for row in waist_df.itertuples():
        if row.n_bones != 1:
            continue
        next_val = row.Index + 1
        last_val = row.Index
        max_val = row.n_bones
        while waist_df.loc[next_val, 'n_bones'] >= waist_df.loc[last_val, 'n_bones']:
            if next_val + 1 > waist_df.index[-1]:
                break
            else:
                max_val = waist_df.loc[next_val, 'n_bones']
                last_val = next_val
                next_val += 1
        if max_val < 3:
            continue
        else:
            incr_ranges[row.Index] = len([i for i in range(row.Index, next_val)])

    if len(incr_ranges) == 0:
        print('No waist found')
        return None
    else:
        max_ix = max(incr_ranges, key=incr_ranges.get)
        waist_range = incr_ranges[max_ix]
        return max_ix, waist_range


def select_waist_measurement(waist_df, max_ix, waist_range):
    """
    Select the waist measurement from the dataframe
    :param waist_df: the dataframe with the waist measurements
    :param max_ix: the index of the waist in the dataframe
    :param waist_range: the range of the waist measurements
    :return: the waist measurement
    """
    waist = waist_df.iloc[max_ix:max_ix + waist_range, :]
    waist_ix = waist[waist['n_bones'] == 3].index.max()
    waist_center = waist_df.loc[waist_ix, 'waist_circumference_cm']
    return waist_center, waist_ix
