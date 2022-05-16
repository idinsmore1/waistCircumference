import numpy as np
import pandas as pd
import utilities as ut
import matplotlib.pyplot as plt
import argparse
import os
from dicomseries import DicomSeries
from skimage import measure

# Wrapper function
def main():
    parser = argparse.ArgumentParser(description='Calculate waist circumference')
    parser.add_argument('-i', '--input', help='Input directory containing the dicom files', required=True)
    parser.add_argument('-o', '--output-dir', help='Output directory', required=True)
    args = parser.parse_args()

    # Get the input directory
    input_dir = args.input

    ct = DicomSeries(input_dir)
    ct_images = ct.read_dicom_series('*', win_center=100, win_width=400)
    ct_images = np.moveaxis(ct_images, -1, 0)
    # Get the output file name
    output_file = f'MRN{ct.mrn}_waist_circumference.csv'
    if args.output_dir[-1] == '/':
        args.output_dir = args.output_dir[:-1]
    output_dir = f'{args.output_dir}/MRN{ct.mrn}/'

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Calculate the waist circumference
    bones_seg = np.stack([ut.threshold_segment_ct_image(ct_image) for ct_image in ct_images])
    # Get the information for each slice
    bone_info = {}
    new_bones = []
    for index, seg in enumerate(bones_seg):
        # print(index)
        lbl = measure.label(seg)
        props = measure.regionprops(lbl)
        for prop in props:
            if prop.area < 80:
                lbl[lbl == prop.label] = 0
        new_bones.append(lbl)
        bone_info[f'{index}'] = {'bones': len(np.unique(lbl)) - 1}
    new_bones = np.stack(new_bones, axis=0)

    # Create a dataframe with the information
    bone_df = pd.DataFrame(bone_info).T
    bone_df['rolling_bones'] = bone_df.bones.rolling(bone_df.shape[0] // 12,
                                                     min_periods=bone_df.shape[0] // 12).mean().round(3)
    # Go 1 cm above and 4.5 cm below the minimum value
    min_ix = np.int64(bone_df.rolling_bones.idxmin())
    thickness = ct.series_info['thickness']
    slices_up = np.ceil(1 / thickness).astype(int)
    slices_down = np.ceil(4.5 / thickness).astype(int) + 1
    possible_circs = bone_df.iloc[(min_ix - slices_up):(min_ix + slices_down), :]
    wcs = []
    for ix in possible_circs.index:
        ix = int(ix)
        wc = ut.get_binary_body(ct_images[ix])
        wc = ut.fill_binary_gaps(wc)
        wc = ut.measure_circumference(wc, ct.series_info['width'])
        wcs.append(wc)
    possible_circs['waist_circ'] = wcs
    possible_circs.to_csv(f'{output_dir}/{output_file}')
    min_slice_wc = possible_circs.waist_circ.idxmin()
    possible_circs.loc[[min_slice_wc]].to_csv(f'{output_dir}/min_{output_file}')
    plt.imsave(f'{output_dir}/min_ix_{output_file.replace(".csv", ".png")}', ct_images[int(min_slice_wc)])
    return possible_circs


if __name__ == '__main__':
    main()