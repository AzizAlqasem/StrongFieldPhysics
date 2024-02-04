"""Calculate the total spectra (counts) for each angle
total_spectra = sum(spectra_i)
total_laser_shots = sum(laser_shots_i)
i is the round number

Example:
#### User input ####
TARGET = 'Xe'
start_round = 1
end_round = 34
exclude_rounds = [] # e.g. [3, 8, 9]

first_angle = -4
last_angle = 90
angle_step = 2
offset_angle = 0

TDC = '2228A'
number_of_tdc_bins = 2048
#### End of user input ####
"""


import numpy as np
import os
import re
# Local modules
from StrongFieldPhysics.parser.data_files import read_header, get_header_info


def combine_scans(target:str, start_round:int, end_round:int, exclude_rounds:list, first_angle:int, \
                  last_angle:int, angle_step:int, offset_angle:int = 0, TDC:str = "2228A", \
                number_of_tdc_bins:int = 2048, save=True, Apply_median_filter = False, median_threshold=30):
    # list all (.csv) files in the current director
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('2228A.csv') and "Round" in f]
    # Load median data
    if Apply_median_filter:
        angle_arr, median_arr, mean_arr = np.loadtxt('median_TDC2228A.dat').T
    # Creat a two dimensional array to store the averged spectra
    # The dimensions are: number of bins, number of angles
    number_of_angles = int((last_angle-first_angle)/angle_step)+1
    total_spectra_arr = np.zeros((number_of_tdc_bins, number_of_angles), dtype=np.int64)
    total_laser_shots_arr = np.zeros(number_of_angles, dtype=np.int64)
    average_count_arr = np.zeros(number_of_angles, dtype=np.float64)

    for f in files:
        # From the file name, read the round number
        round_n = re.findall(r'Round(\d+)_', f)
        round_n = int(round_n[0])
        # Skip the rounds in the exclude_rounds list and outside the start_round and end_round range
        if round_n in exclude_rounds or round_n < start_round or round_n > end_round:
            continue
        headrs = read_header(f) # read the header list from the file
        # From the header list, read the experment name  and find the angle
        ang = re.findall(r'ang(-?\d+)_', f)
        ang = int(ang[0])
        average_count = get_header_info(headrs, 'Average hit/shot')
        average_count = float(average_count)
        if Apply_median_filter:
            # Find the median of the average hit/shot
            median = median_arr[angle_arr == ang]
            # mean = mean_arr[np.where(angle_arr == ang)]
            # Exclude the rounds with average hit/shot far from the median
            # print(f"average_count = {average_count}, median = {round(median,5)}, diff = {round(abs(average_count - median)/median*100, 2)}%, angle = {ang}")
            if abs(average_count - median)/median > median_threshold/100 :
                print(f"Warning! The file {f} is excluded because the average hit/shot is far from the median")
                print(f"average_count = {average_count}, median = {median}, diff = {abs(average_count - median)/median*100}%")
                continue
        # read the file
        data = np.loadtxt(f, delimiter=',')
        # second column is the yield
        yield_ = data[:,1].astype(np.int64)

        # The ang_index has to be positive and increase by 1
        ang_index = int((ang-first_angle)/angle_step) # index of the angle in the total_spectra_arr
        # From the header list, read the average hit/shot
        # From the header list, read the laser shots
        laser_shots = get_header_info(headrs, 'Total Laser shot')
        laser_shots = int(laser_shots)

        # Calculate the total spectra
        total_spectra_arr[:, ang_index] += yield_

        # Store the total laser shots and average count for this ang_indexle and round
        average_count_arr[ang_index] += average_count * laser_shots
        total_laser_shots_arr[ang_index] += laser_shots

    # Step 2: divide by the total number of laser shots
    average_count_arr = average_count_arr / total_laser_shots_arr

    if save:
        fn = f'{target}_total_spectra'
        fn2 = f'{target}_average_count'
        if Apply_median_filter:
            fn += f'_median{int(median_threshold)}'
            fn2 += f'_median{int(median_threshold)}'
        # Save the averaged spectra to a file
        header = 'Angle (Columns): '+str(first_angle)+' to '+str(last_angle)+' step '+str(angle_step) + " degrees"  +'\nOffset angle: '+str(offset_angle)+'\nTDC bins (Rows): '+str(number_of_tdc_bins)
        np.savetxt(f'{fn}_TDC2228A.dat', total_spectra_arr, fmt='%d', header=header, delimiter='\t')

        # Save the angle, averaged count and total laser shots to a file
        scaned_angles = np.arange(first_angle, last_angle+angle_step, angle_step) + offset_angle
        np.savetxt(f'{fn2}s_TDC2228A.dat', np.c_[scaned_angles, average_count_arr, total_laser_shots_arr], fmt=['%d','%.16f', "%d"], header='Angle | Avg. count | Total Laser shot (k)', delimiter='\t\t')

    return total_spectra_arr, average_count_arr, total_laser_shots_arr