import numpy as np
def calculate_ate(true_x, true_y, estimated_x, estimated_y):
    error = np.sqrt((true_x - estimated_x)**2 + (true_y - estimated_y)**2)
    ate = np.mean(error)
    return ate

def calculate_rte(true_x, true_y, estimated_x, estimated_y):
    true_diff_x = np.diff(true_x)
    true_diff_y = np.diff(true_y)
    estimated_diff_x = np.diff(estimated_x)
    estimated_diff_y = np.diff(estimated_y)
    error = np.sqrt((true_diff_x - estimated_diff_x)**2 + (true_diff_y - estimated_diff_y)**2)
    rte = np.mean(error) 
    return rte