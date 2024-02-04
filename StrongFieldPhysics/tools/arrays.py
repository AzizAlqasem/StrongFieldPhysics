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