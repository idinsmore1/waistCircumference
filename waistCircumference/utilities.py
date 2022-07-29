from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numpy import ndarray
from skimage.measure import label, perimeter, regionprops


def imshow(*args, **kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage: 
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title = kwargs.get('title', '')
    save_im = kwargs.get('save_im', None)
    if len(args) == 0:
        raise ValueError("No images given to imshow")
    elif len(args) == 1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n = len(args)
        if type(cmap) == str:
            cmap = [cmap] * n
        if type(title) == str:
            title = [title] * n
        plt.figure(figsize=(n * 5, 10))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
    if save_im is not None:
        print('saving img')
        plt.savefig(f'{save_im}', facecolor='w')
    plt.show()


# General utilities for both bone segmentation and waist measurement
@njit
def get_binary_body(pixel_array: np.ndarray) -> np.ndarray:
    """
    Binarize a CT scan between air and body
    :param pixel_array: dicom pixel array
    :return: binary image as numpy array
    """
    binary_im = pixel_array > np.min(pixel_array)
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


# Bone segmentation utilities
def get_largest_connected_component(binary_image: np.ndarray) -> np.ndarray:
    """
    A function to get the largest connected component from binary image (in this case it should be the body)
    :param binary_image: the output from binarize_image
    :return: an numpy array containing only the largest connected component
    """
    labels = label(binary_image)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]  # the 0 label is by default background so take the rest
    largest = max(list_seg, key=lambda x: x[1])[0]
    labels_max = (labels == largest).astype(int)
    return labels_max


# @njit
def remove_exterior_artifacts(ct_image: np.ndarray, largest_connected_component: np.ndarray) -> np.ndarray:
    """
    A function to remove artifacts outside the body from CT scans
    :param ct_image: the original ct image
    :param largest_connected_component: the output from get_largest_connected_component
    :return: the cleaned up original image
    """
    ct_image[largest_connected_component != 1] = np.min(ct_image)
    return ct_image


# @njit
def threshold_image(ct_image: np.ndarray, lower_bound: np.ndarray = 175) -> np.ndarray:
    """
    A function to threshold an image for bone segmentation
    :param ct_image: the ct image with exterior artifacts removed
    :param lower_bound: the lower bound of hounsfield units, anything below this will be set to zero
    :return: a thresholded numpy array
    """
    ct_new = ct_image.copy()
    ct_new[ct_new < lower_bound] = 0
    ct_new[ct_new > 0] = 1
    return ct_new


def fill_bone_gaps(threshold_im: np.ndarray) -> np.ndarray:
    """
    A function to fill in gaps in the bone by separating them from the background
    :param threshold_im: the output of threshold image
    :return: a numpy array with filled gaps
    """
    filled = label(threshold_im, background=1, connectivity=2)
    filled[filled != 1] = 2
    filled = filled - 1
    return filled


def threshold_segment_ct_image(ct_image: np.ndarray) -> np.ndarray:
    """Function to perform raw threshold segmentation, to move into 2D adaptive segmentation
    :param ct_image: the raw ct image
    :return
    """
    im = ct_image.copy()
    binary_im = get_binary_body(im) # jitted
    body = get_largest_connected_component(binary_im)
    new_im = remove_exterior_artifacts(ct_image, body)
    bone_seg = threshold_image(new_im) # jitted
    bone_seg = fill_bone_gaps(bone_seg)
    return bone_seg


def get_bones_set(filled_im: np.ndarray) -> set[tuple]:
    """
    A function to get the set of pixels marked as bone from a filled image
    :param filled_im: the output of fill_bone_gaps
    :return: a set containing pixel tuples of bones
    """
    bones = {tuple(pixel) for pixel in np.argwhere(filled_im == 1)}
    return bones


def get_non_bones_set(filled_im: np.ndarray) -> set[tuple]:
    """
    A function to get the set of pixels not marked as bone from a filled image
    :param filled_im: the output of fill_bone_gaps
    :return: a set containing pixel tuples of non-bones
    """
    non_bones = {tuple(pixel) for pixel in np.argwhere(filled_im == 0)}
    return non_bones


def get_planar_neighbors(bone: tuple, shape: tuple = (512, 512)) -> list:
    """
    A function to get the indices of 8-connected neighbors of a pixel within a specified shaped image
    :param bone: a pixel coordinate tuple from bones set
    :param shape: the shape of the image
    :return: a list of valid indices to check
    """
    # print(bone)
    y, x = bone
    ymax, ymin = (shape[0] - 1, 0)
    xmax, xmin = (shape[1] - 1, 0)
    neighbor_ixs = [
        [y + 1, x],
        [y - 1, x],
        [y, x + 1],
        [y, x - 1],
        [y + 1, x + 1],
        [y + 1, x - 1],
        [y - 1, x + 1],
        [y - 1, x - 1]
    ]
    drop_ixs = []
    for i, ix in enumerate(neighbor_ixs):
        new_y, new_x = ix
        if new_y > ymax or new_y < ymin or new_x > xmax or new_x < xmin:
            drop_ixs.append(i)
    valid_indices = [ele for idx, ele in enumerate(neighbor_ixs) if idx not in drop_ixs]
    return valid_indices


def find_2d_boundaries(bones_set: set, filled_im: np.ndarray) -> set[tuple]:
    """
    A function to get the boundary pixels from a 2D filled image
    :param bones_set: the output of get_bones_set (set of pixel pairs that are marked as bone)]
    :param filled_im: the output of fill_bone_gaps
    :return: a set containing pixel values of the boundary pixels of the bones
    """
    boundaries = set()
    for bone in bones_set:
        conditions = get_planar_neighbors(bone, filled_im.shape)
        neighbors = np.array([filled_im[condition[0], condition[1]] for condition in conditions])
        # If any of the neighbors are not bone, mark this as a boundary
        if np.any(neighbors == 0):
            boundaries.add(bone)
    return boundaries


def create_boundary_boxes(ct_image: np.ndarray, filled_im: np.ndarray, boundary_pixel: tuple, pixel_radius: int):
    """
    A function to create bounding boxes around the boundary pixels of interest
    :param ct_image: the original HU image
    :param filled_im: the output from fill_bone_gaps
    :param boundary_pixel: a pixel from find_2d_boundaries
    :param pixel_radius: the pixel radius to make the bounding box (half of the width/length)
    :return: the image and label bounding boxes, and the pixel location of the boundary pixel
    """
    by, bx = boundary_pixel
    win_ymin = by - pixel_radius
    win_ymax = by + pixel_radius + 1
    win_xmin = bx - pixel_radius
    win_xmax = bx + pixel_radius + 1

    # Check if the window is out of bounds
    ymax, xmax = ct_image.shape[0] - 1, ct_image.shape[1] - 1

    # Keep track of the boundary pixel
    pixel_loc = [pixel_radius, pixel_radius]

    if win_ymin < 0:
        win_ymin = 0
        pixel_loc[0] = by
    if win_ymax > ymax:
        win_ymax = ymax + 1
        pixel_loc[0] = pixel_radius
    if win_xmin < 0:
        win_xmin = 0
        pixel_loc[1] = bx
    if win_xmax > xmax:
        win_xmax = xmax + 1
        pixel_loc[1] = pixel_radius

    ct_bd_box = ct_image[win_ymin:win_ymax, win_xmin:win_xmax]
    # print(ct_bd_box.shape)
    lbl_bd_box = filled_im[win_ymin:win_ymax, win_xmin:win_xmax]
    # print(lbl_bd_box.shape)
    return ct_bd_box, lbl_bd_box, tuple(pixel_loc)


def calculate_probability(arr, mean, var):
    """
    A function to calculate the Gaussian probability of a pixel belonging to a specific distribution
    :param arr: the pixel array to calculate probabilities
    :param mean: the mean of the class distribution
    :param var: the variance of the class distribution
    :return: an array of the same shape as arr with probabilities
    """
    exponent = np.exp(-((arr - mean) ** 2 / (2 * var)))
    return (1 / (np.sqrt(2 * np.pi) * np.sqrt(var))) * exponent


def get_decision_array(ct_bd_box: np.ndarray, lbl_bd_box: np.ndarray) -> np.ndarray:
    """
    A function to get the decision of belonging to the bone class or not
    :param ct_bd_box: the bounding box of the original HU image
    :param lbl_bd_box: the bounding box of the filled image
    :return: a boolean array containing the new class designations
    """
    # Get pixels belonging to each class
    bd_bones = np.argwhere(lbl_bd_box == 1)
    bd_non_bone = np.argwhere(lbl_bd_box == 0)

    # Get the mean and variances for each class
    # 2D decision array means and variances
    if len(bd_bones[0]) == 2:
        bone_mean = np.array([ct_bd_box[pt[0], pt[1]] for pt in bd_bones]).mean()
        bone_var = np.array([ct_bd_box[pt[0], pt[1]] for pt in bd_bones]).var()
        non_bone_mean = np.array([ct_bd_box[pt[0], pt[1]] for pt in bd_non_bone]).mean()
        non_bone_var = np.array([ct_bd_box[pt[0], pt[1]] for pt in bd_non_bone]).var()
    # 3D decision array means and variances
    elif len(bd_bones[0]) == 3:
        bone_mean = np.array([ct_bd_box[pt[0], pt[1], pt[2]] for pt in bd_bones]).mean()
        bone_var = np.array([ct_bd_box[pt[0], pt[1], pt[2]] for pt in bd_bones]).var()
        non_bone_mean = np.array([ct_bd_box[pt[0], pt[1], pt[2]] for pt in bd_non_bone]).mean()
        non_bone_var = np.array([ct_bd_box[pt[0], pt[1], pt[2]] for pt in bd_non_bone]).var()
    else:
        print("Error: wrong number of dimensions")
        return None

    if bone_var == 0 or non_bone_var == 0:
        return lbl_bd_box
    # Calculate the probability for each pixel to belong to each class
    bone_prob = calculate_probability(ct_bd_box, bone_mean, bone_var)
    non_bone_prob = calculate_probability(ct_bd_box, non_bone_mean, non_bone_var)

    # bone_reclass is True if classified as bone, False otherwise
    bone_reclass = bone_prob > non_bone_prob
    return bone_reclass


def adaptive_2d_bone_thresholding(ct_image: np.ndarray, filled_im: np.ndarray, bones: set,
                                  boundaries: set, px_radius: int) -> \
        tuple[ndarray, set[Any], set[Any]]:
    """
    A wrapper function to perform adaptive 2D bone thresholding on a ct image
    :param ct_image: the Hounsfield unit image
    :param filled_im: the output from fill_bone_gaps
    :param bones: set containing all pixels classified as bone
    :param non_bones: set containing all pixels classified as not bone
    :param boundaries: the output from find 2d boundaries
    :param px_radius: the pixel radius to make the bounding box (half of the width/length)
    :return: a numpy array with the threshold image
    """
    new_labels = filled_im.copy()
    while True:
        errors = set()
        for boundary in boundaries:
            ct_bd_box, lbl_bd_box, pixel_loc = create_boundary_boxes(ct_image, new_labels, boundary, px_radius)
            reclassed = get_decision_array(ct_bd_box, lbl_bd_box)
            # print(f'CT Shape = {ct_bd_box.shape}, LBL Shape = {lbl_bd_box.shape}, pixel_loc = {pixel_loc}')
            if reclassed[pixel_loc[0], pixel_loc[1]] != lbl_bd_box[pixel_loc[0], pixel_loc[1]]:
                errors.add(boundary)

        if len(errors) == 0:
            break
        else:
            # print(len(errors))
            bones = bones.difference(errors)
            # non_bones = non_bones.difference(errors)
            new_labels = np.zeros(filled_im.shape)
            for bone in bones:
                new_labels[bone[0], bone[1]] = 1
            boundaries = find_2d_boundaries(bones, new_labels)
    return new_labels, bones


def get_3d_neighbors(bone: tuple) -> list:
    """
    A function to get the planar neighbors and 3d neighbors of a pixel
    :param bone: a 3d pixel coordinate tuple from bones set
    :return: a list of 3d neighbors
    """
    y, x, z = bone
    neighbor_ixs = get_planar_neighbors((y, x))
    neighbor_ixs = [[ix[0], ix[1], z] for ix in neighbor_ixs]
    if z != 0:
        neighbor_ixs.append([y, x, z - 1])
    if z != 511:
        neighbor_ixs.append([y, x, z + 1])
    return neighbor_ixs


def find_3d_boundaries(bones_set: set, segmented_volume: np.ndarray) -> set[tuple]:
    boundaries = set()
    for bone in bones_set:
        conditions = get_3d_neighbors(bone)
        neighbors = np.array([segmented_volume[condition[0], condition[1], condition[2]] for condition in conditions])
        if np.any(neighbors == 0):
            boundaries.add(bone)
    return boundaries


def create_3d_boundary_boxes(ct_volume: np.ndarray, seg_volume: np.ndarray, boundary_pixel: tuple,
                             pixel_radius: int):
    """
    Create the 3d boundary box around the boundary pixel
    :param ct_volume: the ct volume in hounsfield units
    :param seg_volume: the volume containing the 2d segmented bones
    :param boundary_pixel: the bone pixel coordinates [y, x, z]
    :param pixel_radius: the radius of the bounding box
    :return: the image and label bounding boxes, and the pixel location of the boundary pixel
    """
    by, bx, bz = boundary_pixel
    win_ymin = by - pixel_radius
    win_ymax = by + pixel_radius + 1
    win_xmin = bx - pixel_radius
    win_xmax = bx + pixel_radius
    win_zmin = bz - 1
    win_zmax = bz + 1
    # Check if the window is out of bounds
    ymax, xmax = ct_volume.shape[0] - 1, ct_volume.shape[1] - 1
    pixel_loc = [pixel_radius, pixel_radius, 1]
    if bz == 0:
        pixel_loc[2] = 0
    if bz == ct_volume.shape[2] - 1:
        pixel_loc[2] = 2
    if win_ymin < 0:
        win_ymin = 0
        pixel_loc[0] = by
    if win_ymax > ymax:
        win_ymax = ymax
        pixel_loc[0] = pixel_radius
    if win_xmin < 0:
        win_xmin = 0
        pixel_loc[1] = bx
    if win_xmax > xmax:
        win_xmax = xmax
        pixel_loc[1] = pixel_radius

    if bz == 0:
        ct_bd_box = ct_volume[win_ymin:win_ymax, win_xmin:win_xmax, bz:bz + 3]
        lbl_bd_box = seg_volume[win_ymin:win_ymax, win_xmin:win_xmax, bz:bz + 3]
    elif bz == ct_volume.shape[2] - 1:
        ct_bd_box = ct_volume[win_ymin:win_ymax, win_xmin:win_xmax, bz - 2:]
        lbl_bd_box = seg_volume[win_ymin:win_ymax, win_xmin:win_xmax, bz - 2:]
    else:
        ct_bd_box = ct_volume[win_ymin:win_ymax, win_xmin:win_xmax, win_zmin:win_zmax]
        lbl_bd_box = seg_volume[win_ymin:win_ymax, win_xmin:win_xmax, win_zmin:win_zmax]
    return ct_bd_box, lbl_bd_box, tuple(pixel_loc)


def adaptive_3d_threshold(ct_volume: np.ndarray, seg_volume: np.ndarray, bones: set, non_bones: set, boundaries: set,
                          pixel_radius: int):
    """
    Adaptive thresholding for the 3d volume
    :param ct_volume: the ct volume in hounsfield units
    :param seg_volume: the volume containing the 2d segmented bones
    :param bones: the set of bone labels
    :param non_bones: the set of non-bone labels
    :param boundaries: the set of boundary labels
    :param pixel_radius: the radius of the bounding box
    :return: a numpy array with the thresholded volume
    """
    new_segmentation = seg_volume.copy()
    while True:
        errors = set()
        for boundary in boundaries:
            ct_bd_box, lbl_bd_box, pixel_loc = create_3d_boundary_boxes(ct_volume, new_segmentation, boundary,
                                                                        pixel_radius)
            reclassed = get_decision_array(ct_bd_box, lbl_bd_box)
            if reclassed[pixel_loc[0], pixel_loc[1], pixel_loc[2]] != lbl_bd_box[
                pixel_loc[0], pixel_loc[1], pixel_loc[2]]:
                errors.add(boundary)
        if len(errors) == 0:
            break
        else:
            print(len(errors))
            bones = bones.difference(errors)
            non_bones = non_bones.difference(errors)
            new_segmentation = np.zeros(seg_volume.shape)
            for bone in bones:
                new_segmentation[bone[0], bone[1], bone[2]] = 1
            boundaries = find_3d_boundaries(bones, new_segmentation)

    return new_segmentation, bones, non_bones


# Waist measurement utilities
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


def detect_number_of_bones(pixel_array, upper_bound=175, area_threshold=200):
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
        try:
            max_ix = max(incr_ranges, key=incr_ranges.get)
        except KeyError:
            print('No waist found')
            return None, None
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
