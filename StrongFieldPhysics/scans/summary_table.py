import numpy as np
import os
import re

from StrongFieldPhysics.parser.data_files import read_header, get_header_info


def summary_table(dir='.', TDC:str = "2228A", save=True, ang_re=r'ang(-?\d+)_', round_re=r'Round(\d+)_'):
    # list all (.csv) files in the current director
    files = [f for f in os.listdir(dir) if os.path.isfile(f) and f.endswith(f'{TDC}.csv') and "Round" in f]
    round_list = []
    angle_list = []
    average_hit_list = []
    laser_shots_list = []
    for f in files:
        # read the file
        data = np.loadtxt(f, delimiter=',')
        # second column to int
        data = data[:,1]
        headrs = read_header(f)
        # save the file as .dat
        #Format float to 4 decimal places
        fmt = '%d'
        # np.savetxt(f[:-4]+'.dat', data, comments='', fmt=fmt,  footer="".join(headrs))

        # From the header list, read the experment name  and find the angle
        ang = re.findall(ang_re, f)
        angle_list.append(ang[0])
        # From the header list, read the average hit/shot
        average_hit = get_header_info(headrs, 'Average hit/shot')
        average_hit = round(float(average_hit), 10)
        average_hit_list.append("{:.12f}".format(average_hit))
        # From the header list, read the laser shots
        laser_shots = get_header_info(headrs, 'Total Laser shot')
        laser_shots_list.append(laser_shots)
        # From the file name, read the round number
        round_n = re.findall(round_re, f)
        round_list.append(round_n[0])

    # Save the angle and average hit/shot to a file
    if save:
        np.savetxt(f'Summary_TDC{TDC}.dat', np.c_[round_list, angle_list, average_hit_list, laser_shots_list], fmt='%s', header='Round # | Angle | Avg. count | Total Laser shot (k)', delimiter='\t')


    #### Find the median of the average hit/shot and exclude the rounds with average hit/shot far from the median
    # convert round_list to int, angle_list to float, average_hit_list to float
    round_list = np.array(round_list).astype(np.int64)
    angle_list = np.array(angle_list).astype(np.float64)
    average_hit_list = np.array(average_hit_list).astype(np.float64)
    # For each angle of angle_list, find the median of the average_hit_list
    median_list = []
    mean_list = []
    angle_list2 = []
    for ang in np.unique(angle_list):
        inds_bool = angle_list==ang
        angle_list2.append(ang)
        median_list.append(np.median(average_hit_list[inds_bool]))
        mean_list.append(np.mean(average_hit_list[inds_bool]))
    # save the median to a file
    if save:
        np.savetxt(f'median_TDC{TDC}.dat', np.c_[angle_list2, median_list, mean_list], fmt=['%.2f','%.16f', "%.16f"], header='Angle | Median | Mean', delimiter='\t\t')