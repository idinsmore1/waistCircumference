import argparse

import pandas as pd

from dicomseries import DicomSeries
from waistCircumference.utilities import binarize_scan, fill_binary_gaps, mark_body, measure_circumference

# This is a sample Python script.
parser = argparse.ArgumentParser(description='Automatically Calculate Waist Circumference from CT Scan')
parser.add_argument('-i', '--input', help='Input DICOM Series Directory', required=True)
parser.add_argument('-o', '--output', help='Output CSV File Directory', required=True)
args = parser.parse_args()


def main():
    dicom_series = DicomSeries(args.input)
    description = f'MRN{dicom_series.mrn}_ACC{dicom_series.acc}_{dicom_series.cut}_waist_circumference'
    outfile = f'{args.output}/{description}.csv'
    ct_scan = dicom_series.read_dicom_series('*', -10, 300)
    waist_circumferences = []
    for image in range(ct_scan.shape[2]):
        binary_im = binarize_scan(ct_scan[:, :, image])
        binary_im = fill_binary_gaps(binary_im)
        body_array = mark_body(binary_im)
        measurement = measure_circumference(body_array, dicom_series.series_info['width'])
        waist_circumferences.append(measurement)
    df = pd.DataFrame(waist_circumferences, columns=['waist_circumference'])
    df.to_csv(outfile)


if __name__ == '__main__':
    main()
