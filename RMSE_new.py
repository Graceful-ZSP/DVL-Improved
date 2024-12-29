import numpy as np
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error


def RMSE(true, predicted, DVL_v):
    true_col_0 = true[:, 0]
    predicted_col_0 = predicted[:, 0]
    DVL_v_col_0 = DVL_v[:, 0]
    rmse_predicted_col_0 = np.sqrt(np.mean((true_col_0 - predicted_col_0) ** 2))
    rmse_DVL_v_col_0 = np.sqrt(np.mean((true_col_0 - DVL_v_col_0) ** 2))
    true_col_1 = true[:, 1]
    predicted_col_1 = predicted[:, 1]
    DVL_v_col_1 = DVL_v[:, 1]
    rmse_predicted_col_1 = np.sqrt(np.mean((true_col_1 - predicted_col_1) ** 2))
    rmse_DVL_v_col_1 = np.sqrt(np.mean((true_col_1 - DVL_v_col_1) ** 2))
    # print(rmse_predicted_col_0)
    # print(rmse_predicted_col_1)
    # print(rmse_DVL_v_col_0)
    # print(rmse_DVL_v_col_1)
    rmse_predicted_col = (rmse_predicted_col_0+rmse_predicted_col_1)/2
    rmse_DVL_v_col = (rmse_DVL_v_col_0+rmse_DVL_v_col_1)/2
    
    rmse_improve = 100 * (1 - (rmse_predicted_col / rmse_DVL_v_col))
    # rmse_improve = 100 * ((rmse_DVL_v_col - rmse_predicted_col )/rmse_DVL_v_col)
    rmse_ls = rmse_DVL_v_col
    rmse_predicted = rmse_predicted_col
    return rmse_ls, rmse_predicted, rmse_improve
    # return rmse_DVL_v_col, rmse_predicted_col, improv


def MAE(true, predicted, DVL_v):
    true_col_0 = true[:, 0]
    predicted_col_0 = predicted[:, 0]
    DVL_v_col_0 = DVL_v[:, 0]
    mae_predicted_col_0 = np.sum(np.abs(true_col_0 - predicted_col_0)) / len(true_col_0)
    mae_DVL_v_col_0 = np.sum(np.abs(true_col_0 - DVL_v_col_0)) / len(true_col_0)
    
    true_col_1 = true[:, 1]
    predicted_col_1 = predicted[:, 1]
    DVL_v_col_1 = DVL_v[:, 1]
    mae_predicted_col_1 = np.sum(np.abs(true_col_1 - predicted_col_1)) / len(true_col_1)
    mae_DVL_v_col_1 = np.sum(np.abs(true_col_1 - DVL_v_col_1)) / len(true_col_1)

    mae_predicted_col = (mae_predicted_col_0+mae_predicted_col_1)/2
    mae_DVL_v_col = (mae_DVL_v_col_0+mae_DVL_v_col_1)/2

    mae_ls =mae_DVL_v_col
    mae_predicted =mae_predicted_col
    mae_improve = 100 * (1 - (mae_predicted / mae_ls))
    return mae_ls, mae_predicted,mae_improve

def NSE_R2(true, predicted, DVL_v):
    true_col_0 = true[:, 0]
    predicted_col_0 = predicted[:, 0]
    DVL_v_col_0 = DVL_v[:, 0]
    true_avg = np.mean(true_col_0)
    temp_dvl0 = np.sum((predicted_col_0 - true_col_0) ** 2) / np.sum((true_avg - true_col_0) ** 2)
    temp_dvl1 = np.sum((DVL_v_col_0 - true_col_0) ** 2) / np.sum((true_avg - true_col_0) ** 2)
    r2_predicted_col_0 = 1 - temp_dvl0
    r2__DVL_v_col_0 = 1 - temp_dvl1

    true_col_1 = true[:, 1]
    predicted_col_1 = predicted[:, 1]
    DVL_v_col_1 = DVL_v[:, 1]
    true_avg = np.mean(true_col_1)
    temp_dvl2 = np.sum((predicted_col_1 - true_col_1) ** 2) / np.sum((true_avg - true_col_1) ** 2) 
    temp_dvl3 = np.sum((DVL_v_col_1 - true_col_1) ** 2) / np.sum((true_avg - true_col_1) ** 2)
    r2_predicted_col_1 = 1 - temp_dvl2
    r2__DVL_v_col_1 = 1 - temp_dvl3

    r2_predicted_col = (r2_predicted_col_0 + r2_predicted_col_1)/2
    r2_DVL_v_col = (r2__DVL_v_col_0 + r2__DVL_v_col_1)/2   

    r2_predicted = r2_predicted_col
    r2_ls = r2_DVL_v_col

    r2_improve = 100 * ((r2_predicted / r2_ls) - 1)
    return r2_ls, r2_predicted,r2_improve

def VAF(true, predicted, DVL_v):

    vaf_DVL_v_cols = []
    vaf_predicted_cols = []

    for col in range(true.shape[1]):
        true_col = true[:, col]
        predicted_col = predicted[:, col]
        DVL_v_col = DVL_v[:, col]

        true_var = np.var(true_col)
        temp_dvl = np.var(DVL_v_col - true_col)
        vaf_DVL_v_col = (1 - temp_dvl / true_var) * 100

        temp_pre = np.var(predicted_col - true_col)
        vaf_predicted_col = (1 - temp_pre / true_var) * 100

        vaf_DVL_v_cols.append(vaf_DVL_v_col)
        vaf_predicted_cols.append(vaf_predicted_col)

    vaf_DVL_v_avg = np.mean(vaf_DVL_v_cols)
    vaf_predicted_avg = np.mean(vaf_predicted_cols)
    vaf_improve = 100 * ((vaf_predicted_avg / vaf_DVL_v_avg) - 1)
    return vaf_DVL_v_avg, vaf_predicted_avg,vaf_improve