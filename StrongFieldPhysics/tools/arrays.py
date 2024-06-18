import numpy as np


# A function that finds index of the closest value in an array and returns the index
def find_index_near(array, value, max_diff=0.1):
    if isinstance(array, list):
        array = np.array(array)
    indx = (np.abs(array - value)).argmin()
    # check if index is at the edge
    if (indx == 0 or indx == len(array) - 1) and np.isclose(abs(array[indx] - value), 0.0) == False :
        # return None
        raise ValueError("The value is at the edge of the array")
    # Check if the value is close enough
    if abs(array[indx] - value) > max_diff:
        # return None
        raise ValueError("The value is not close enough to the array values")
    return indx



def find_index_near2(array, value, max_diff=0.1): # same as find_index_near but without the edge check
    if isinstance(array, list):
        array = np.array(array)
    indx = (np.abs(array - value)).argmin()
    # Check if the value is close enough
    if abs(array[indx] - value) > max_diff:
        # return None
        raise ValueError("The value is not close enough to the array values")
    return indx


# Function that you give it an array of data, and an array of specific values,
# and it returns the index of the closest value in the data array for each value in the specific values array
def find_index_near_array(data, values, max_diff=0.1):
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(values, list):
        values = np.array(values)
    return [find_index_near(data, value, max_diff) for value in values]


# function that return the minimum index of an array looking at specific range of th array
def find_min_in_range(array, start, end, max_diff=0.1):
    """ find minimum value within a specific range of an array
    """
    idx1 = find_index_near(array, start, max_diff)
    idx2 = find_index_near(array, end, max_diff)
    return np.argmin(array[idx1:idx2]) + idx1

def find_max_in_range(array, start, end, max_diff=0.1):
    """ find maximum value within a specific range of an array
    """
    idx1 = find_index_near(array, start, max_diff)
    idx2 = find_index_near(array, end, max_diff)
    return np.argmax(array[idx1:idx2]) + idx1

# function that find min in y_arr that crossponds to a specific range of x_arr
def find_indx_min_in_yarr_from_xrange(x_arr, y_arr, start, end, max_diff=0.1):
    """ find minimum value of y_arr within a specific range within x_arr
    """
    idx1 = find_index_near2(x_arr, start, max_diff)
    idx2 = find_index_near2(x_arr, end, max_diff)
    idx_min = np.argmin(y_arr[idx1:idx2])
    return idx1 + idx_min

def find_indx_max_in_yarr_from_xrange(x_arr, y_arr, start, end, max_diff=0.1):
    """ find maximum value of y_arr within a specific range within x_arr
    """
    idx1 = find_index_near2(x_arr, start, max_diff)
    idx2 = find_index_near2(x_arr, end, max_diff)
    idx_max = np.argmax(y_arr[idx1:idx2])
    return idx1 + idx_max

def get_element_indx_of_list_of_array(list_of_array, index):
    """ get the index of an element in a list of array
    eg., list_of_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    get_element_indx_of_list_of_array(list_of_array, 1) -> [2, 5, 8]
    """
    return [arr[index] for arr in list_of_array]