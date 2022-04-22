import argparse

import pandas as pd
import os
from dicomseries import DicomSeries
from utilities import (
    binarize_scan,
    fill_binary_gaps,
    mark_body,
    measure_circumference,
    detect_number_of_bones,
    get_waist_range,
    select_waist_measurement
)

# This is a sample Python script.
parser = argparse.ArgumentParser(description='Automatically Calculate Waist Circumference from CT Scan')
parser.add_argument('-i', '--input', help='Input DICOM Series Directory', required=True)
parser.add_argument('-o', '--output', help='Output CSV File Directory', required=True)
parser.add_argument('-t', '--threshold', help='Bone Area Threshold to filter number of bones', required=False,
                    default=100)
args = parser.parse_args()


def main():
    # Create a DicomSeries object
    dicom_series = DicomSeries(args.input)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # Make the base name of the output file
    description = f'MRN{dicom_series.mrn}_ACC{dicom_series.acc}_{dicom_series.cut}_waist_circumferences'
    # Read the DicomSeries object at an HU where bone is easily visible
    ct_scan = dicom_series.read_dicom_series('*', 0, 500)
    waist_circumferences = {}
    # For each cut, binarize the image, fill the gaps, measure the circumference, and get number of bones
    for image in range(ct_scan.shape[2]):
        image_slc = ct_scan[:, :, image]
        binary_im = binarize_scan(image_slc)
        binary_im = fill_binary_gaps(binary_im)
        body_array = mark_body(binary_im)
        measurement = measure_circumference(body_array, dicom_series.series_info['width'])
        n_bones = detect_number_of_bones(image_slc, upper_bound=250, area_threshold=int(args.threshold))
        waist_circumferences[image] = {'waist_circumference_cm': measurement, 'n_bones': n_bones}
    # Write the results to a CSV file
    df = pd.DataFrame(waist_circumferences).T
    df.to_csv(f'{args.output}/{description}.csv')
    # Select the waist measurement from the CSV file
    max_ix, waist_range = get_waist_range(df)
    print(max_ix, waist_range)
    waist_center, waist_ix, = select_waist_measurement(df, max_ix, waist_range)
    # Store the data around the waist to calculate the mean and standard deviation
    five_measure = df.loc[(waist_ix - 2):(waist_ix + 2), 'waist_circumference_cm']
    fifteen_measure = df.loc[(waist_ix - 8):(waist_ix + 8), 'waist_circumference_cm']
    important_vals = [
        dicom_series.mrn,
        dicom_series.series_info['scan_date'],
        waist_ix,
        waist_center,
        five_measure.mean(),
        five_measure.std(),
        fifteen_measure.mean(),
        fifteen_measure.std()
    ]
    # Write the important values to a CSV file
    pd.DataFrame([important_vals],
                 columns=['MRN', 'ScanDate', 'WaistIndex', 'WaistCenter', '5-Mean', '5-Std', '15-Mean',
                          '15-Std']).to_csv(f'{args.output}/{description}_measurement.csv', index=False)


if __name__ == '__main__':
    main()
