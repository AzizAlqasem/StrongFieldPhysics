# Intensity calibration based on the Photoelectron structure
# At gamma < 1, the 2Up knee is used
# At gamma ~ 1, the 10Up knee is used

import numpy as np
import matplotlib.pyplot as plt


# For 2Up knee
# There exist two lanes before and after the 2Up knee.
# Method:
# 1. Guess the 2Up knee energy position
# 2. Find the slope before the 2Up knee and after the 2Up knee
# 3. Find the intersection point of the two lines and use it as the new 2Up knee energy position
# 4. The quility of the new 2Up knee energy position is determined by two lines fitting quality
# 5. Repeat the process until the new 2Up knee energy position converges or the number of iterations exceeds a certain number
# return the new 2Up knee energy position and use it to calibrate the intensity

def find_2Up_knee(energy_arr, count_arr, E, line1_size, line2_size, line_spaceing, lines_ext=20):
    # find the index of the 2Up knee
    E_2Up_i = np.argmin(np.abs(energy_arr - E))
    # line1
    first_index = E_2Up_i - line1_size - line_spaceing
    last_index = E_2Up_i - line_spaceing
    E_line1 = energy_arr[first_index:last_index]
    count_line1 = count_arr[first_index:last_index]
    line1, residual1, *_ = np.polyfit(E_line1, count_line1, 1, full=True)
    E_line1_arr = energy_arr[first_index:last_index+lines_ext]
    line1_arr = line1[0] * E_line1_arr + line1[1]
    # line2
    first_index = E_2Up_i + line_spaceing
    last_index = E_2Up_i + line2_size + line_spaceing
    E_line2 = energy_arr[first_index:last_index]
    count_line2 = count_arr[first_index:last_index]
    line2, residual2, *_ = np.polyfit(E_line2, count_line2, 1, full=True)
    E_line2_arr = energy_arr[first_index-lines_ext:last_index]
    line2_arr = line2[0] * E_line2_arr + line2[1]
    # find the new 2Up knee energy position
    E_intercept = (line2[1] - line1[1]) / (line1[0] - line2[0])
    line1_sigma = np.sqrt(residual1 / len(E_line1))
    line2_sigma = np.sqrt(residual2 / len(E_line2))
    return E_line1_arr, line1_arr, E_line2_arr, line2_arr, E_intercept, line1_sigma, line2_sigma

def Find_intersection_of_two_fit_lines(energy_arr, count_arr, line1_E, line2_E, line1_size, line2_size, lines_ext=20, E_is_center=True):
    # find the index of the 2Up knee
    line1_E_i = np.argmin(np.abs(energy_arr - line1_E))
    line2_E_i = np.argmin(np.abs(energy_arr - line2_E))
    # line1
    if E_is_center:
        first_index = int(line1_E_i - line1_size/2)
        last_index = int(line1_E_i + line1_size/2)
    else:
        first_index = int(line1_E_i - line1_size)
        last_index = line1_E_i
    E_line1 = energy_arr[first_index:last_index]
    count_line1 = count_arr[first_index:last_index]
    # line1, residual1, *_ = np.polyfit(E_line1, count_line1, 1, full=True)
    line1, cov1 = np.polyfit(E_line1, count_line1, 1, cov=True)
    var_slope1, var_intercept1 = np.diag(cov1)
    E_line1_arr = energy_arr[first_index:last_index+lines_ext]
    line1_arr = line1[0] * E_line1_arr + line1[1]
    # line2
    if E_is_center:
        first_index = int(line2_E_i - line2_size/2)
        last_index = int(line2_E_i + line2_size/2)
    else:
        first_index = line2_E_i
        last_index = int(line2_E_i + line2_size)
    E_line2 = energy_arr[first_index:last_index]
    count_line2 = count_arr[first_index:last_index]
    # line2, residual2, *_ = np.polyfit(E_line2, count_line2, 1, full=True)
    line2, cov2 = np.polyfit(E_line2, count_line2, 1, cov=True)
    var_slope2, var_intercept2 = np.diag(cov2)
    E_line2_arr = energy_arr[first_index-lines_ext:last_index]
    line2_arr = line2[0] * E_line2_arr + line2[1]
    # find the new 2Up knee energy position
    E_intercept = (line2[1] - line1[1]) / (line1[0] - line2[0])
    y_intercept = line1[0] * E_intercept + line1[1]
    E_intercept_error = np.sqrt(((var_intercept1 + var_intercept2) / (line1[1] - line2[1])**2) + ((var_slope1 + var_slope2) / (line1[0] - line2[0])**2))
    # line1_sigma = np.sqrt(residual1 / len(E_line1))
    # line2_sigma = np.sqrt(residual2 / len(E_line2))
    return E_line1_arr, line1_arr, E_line2_arr, line2_arr, E_intercept, y_intercept, E_intercept_error#line1_sigma, line2_sigma


def find_knee(energy_arr, count_arr, E_start, E_end, E_step, line_size):
    # Kind of works but not very reliable yet. more testing is needed
    count_arr = np.log10(count_arr)
    E_range = np.arange(E_start, E_end, E_step)
    line_sigma = np.empty_like(E_range) # sigma is the standard deviation
    for i, E in enumerate(E_range):
        E_2Up_i = np.argmin(np.abs(energy_arr - E))
        first_index = E_2Up_i - line_size//2
        last_index = E_2Up_i + line_size//2 + 1
        line, residual, *_ = np.polyfit(energy_arr[first_index:last_index], count_arr[first_index:last_index], 1, full=True)
        # find the new 2Up knee energy position
        line_sigma[i] = np.sqrt(residual / (line_size+1))
    return E_range, line_sigma


##### The following code is tested to work but failed to achive the designed goal #####
# def find_2Up_knee2(energy_arr, count_arr, E_start, E_end, E_step, line1_size, line2_size, line_spaceing):
#! one way to improve the code is to use interpolation.
#     count_arr = np.log10(count_arr)
#     E_range = np.arange(E_start, E_end, E_step)
#     E_intercept = np.empty_like(E_range)
#     line1_sigma = np.empty_like(E_range) # sigma is the standard deviation
#     line2_sigma = np.empty_like(E_range)
#     for i, E in enumerate(np.arange(E_start, E_end, E_step)):
#         # find the index of the 2Up knee
#         E_2Up_i = np.argmin(np.abs(energy_arr - E))
#         # line1
#         first_index = E_2Up_i - line1_size - line_spaceing
#         last_index = E_2Up_i - line_spaceing
#         print(energy_arr[first_index], energy_arr[last_index])
#         line1, residual1, *_ = np.polyfit(energy_arr[first_index:last_index], count_arr[first_index:last_index], 1, full=True)
#         # line2
#         first_index = E_2Up_i + line_spaceing
#         last_index = E_2Up_i + line2_size + line_spaceing
#         print(energy_arr[first_index], energy_arr[last_index])
#         print('\n')
#         line2, residual2, *_ = np.polyfit(energy_arr[first_index:last_index], count_arr[first_index:last_index], 1, full=True)
#         # find the new 2Up knee energy position
#         E_intercept[i] = (line2[1] - line1[1]) / (line1[0] - line2[0])
#         line1_sigma[i] = residual1 #np.sqrt(residual1 / line1_size)
#         line2_sigma[i] = residual2#np.sqrt(residual2 / line2_size)
#     return E_range, E_intercept, line1_sigma, line2_sigma

