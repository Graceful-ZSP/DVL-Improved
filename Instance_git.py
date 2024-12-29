import torch
import os
import numpy as np
import pandas as pd
from VelocityNet1 import VelocityNet1
from RMSE_new import RMSE,MAE,NSE_R2,VAF
from numpy import load
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
def integrate_velocity(velocity, time_step):
    return np.cumsum(velocity) * time_step

if __name__ == '__main__': 
    T = 20
    batch_size = 16
    flag = '51'
    mode = 'test'  
    path = os.getcwd()
    path = os.path.abspath(os.path.join(path, os.pardir))  
    IMU_in = load(path + '/DVL_VelocityNet/test_combined_data_imu.npy')    
    DVL_V = load(path + '/DVL_VelocityNet/test_combined_data_dvl.npy')  
    GPS_V = load(path + '/DVL_VelocityNet/test_data_gps.npy')
    DVL_V = DVL_V.T
    GPS_V = GPS_V.T
    mean_acc1 = np.mean(IMU_in[0,:,:], axis=0)
    std_acc1 = np.std(IMU_in[0,:,:], axis=0)
    mean_acc2 = np.mean(IMU_in[1,:,:], axis=0)
    std_acc2 = np.std(IMU_in[1,:,:], axis=0)
    mean_acc3 = np.mean(IMU_in[2,:,:], axis=0)
    std_acc3 = np.std(IMU_in[2,:,:], axis=0)  
    mean_dvl = np.mean(DVL_V, axis=0)
    std_dvl = np.std(DVL_V, axis=0) 
    mean_gps = np.mean(GPS_V, axis=0)
    std_gps  = np.std(GPS_V, axis=0)   
    IMU_in[0,:,:] = (IMU_in[0,:,:] - mean_acc1) / std_acc1
    IMU_in[1,:,:] = (IMU_in[1,:,:] - mean_acc2) / std_acc2
    IMU_in[2,:,:] = (IMU_in[2,:,:] - mean_acc3) / std_acc3
    DVL_V_normalized = (DVL_V - mean_dvl) / std_dvl
    GPS_V_normalized = (GPS_V - mean_gps) / std_gps    
    X_gyro = np.zeros((len(IMU_in[0, :, 0]) // T, 3, T))
    X_acc = np.zeros((len(IMU_in[0, :, 0]) // T, 3, T))
    DVL = np.zeros((len(IMU_in[0, :, 0]) // T, 2))
    GPS = np.zeros((len(IMU_in[0, :, 0]) // T, 2))
    n = 0    
    for t in range(0, len(IMU_in[0, :, 0]) - 1, T):
        x_acc = IMU_in[:, t:t + T, 0]
        X_acc[n, :, :] = x_acc[:, :]
        x_gyro = IMU_in[:, t:t + T, 1]
        X_gyro[n, :, :] = x_gyro[:, :]
        y = DVL_V_normalized[n, :]
        DVL[n, :] = y
        z = GPS_V_normalized[n, :]
        GPS[n, :] = z
        n = n + 1
    N1 = len(IMU_in[0, :, 0]) // T
    X_test_acc = torch.from_numpy(X_acc[:, :, :].astype(np.float32))
    X_test_gyro = torch.from_numpy(X_gyro[:, :, :].astype(np.float32))
    X_test_DVL = torch.from_numpy(DVL[:, :].astype(np.float32))
    y_test = torch.from_numpy(GPS[:, :].astype(np.float32))
    X_test_acc = torch.utils.data.DataLoader(dataset=X_test_acc, batch_size=batch_size)
    X_test_gyro = torch.utils.data.DataLoader(dataset=X_test_gyro, batch_size=batch_size)
    X_test_DVL = torch.utils.data.DataLoader(dataset=X_test_DVL, batch_size=batch_size)
    y_test = torch.utils.data.DataLoader(dataset=y_test, batch_size=batch_size)
    V = DVL_V
    model = VelocityNet1()
    model.load_state_dict(torch.load(path + '/DVL_VelocityNet/weighting16.pkl'))  
    print("Last Layer Weights (FC_output):")
    print(model.FC_output[0].weight.data)
    print("Last Layer Bias (FC_output):")
    print(model.FC_output[0].bias.data)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        validation_predictions = []
        for inputs11, inputs12, inputs2, targets in zip(X_test_acc, X_test_gyro, X_test_DVL, y_test):
            inputs11, inputs12, inputs2, targets = inputs11.to(device), inputs12.to(device), inputs2.to(
                device), targets.to(device)
            validation_predictions.append(model(inputs11, inputs12, inputs2).cpu().numpy())

    validation_predictions = np.asarray(validation_predictions, dtype=object)
    validation_predictions = np.concatenate(validation_predictions, axis=0)
    validation_predictions_unnormalized = validation_predictions * std_gps + mean_gps
    predicted_v = np.zeros((N1, 2))
    DVL_v = np.zeros((N1, 2))
    gt_v = np.zeros((N1, 2))
    for i in range(N1):
        predicted_v[i, :] = validation_predictions_unnormalized[i, :]
        DVL_v[i, :] = V[i, :]
        gt_v[i, :] = GPS_V[i, :]
    df = pd.DataFrame(predicted_v)        
    if flag == '42' :
        start_index = 0
        end_index = 798
    elif flag == '50' :
        start_index = 798
        end_index = 1762
    elif flag == '43' :
        start_index = 1762
        end_index = 3558
    elif flag == '51' :
        start_index = 3558
        end_index = 5012
    rmse_ls, rmse_predicted, rmse_improve = RMSE(gt_v[start_index:end_index], predicted_v[start_index:end_index], DVL_v[start_index:end_index])
    mae_ls, mae_predicted,mae_improve = MAE(gt_v[start_index:end_index], predicted_v[start_index:end_index], DVL_v[start_index:end_index])
    r2_ls, r2_predicted,r2_improve = NSE_R2(gt_v[start_index:end_index], predicted_v[start_index:end_index], DVL_v[start_index:end_index])
    vaf_ls, vaf_predicted,vaf_improve = VAF(gt_v[start_index:end_index], predicted_v[start_index:end_index], DVL_v[start_index:end_index])
    df = pd.DataFrame(
        np.array([[rmse_predicted, mae_predicted, r2_predicted, vaf_predicted], [rmse_ls, mae_ls, r2_ls, vaf_ls],[rmse_improve,mae_improve,r2_improve,vaf_improve]]),
        pd.Index([
            'Network prediction', 'Least Squares solution','Imporvement']), columns=['RMSE', 'MAE', 'NSE', 'VAF'])
    print('BeamsNetV1 and Least Squares Results: ')
    print(df)    
    N2 = predicted_v.shape[0]
    if mode == 'test':
        if flag == '42' :
            start_index = 0
            end_index = 798
            df1 = pd.read_csv('test_20240612_134642_dvl.csv')
            df2 = pd.read_csv('test_20240612_134642.csv')
        elif flag == '50' :
            start_index = 798
            end_index = 1762
            df1 = pd.read_csv('test_20240612_142950_dvl.csv')   
            df2 = pd.read_csv('test_20240612_142950.csv')
        elif flag == '43' :
            start_index = 1762
            end_index = 3558
            df1 = pd.read_csv('test_20240612_143443_dvl.csv')
            df2 = pd.read_csv('test_20240612_143443.csv')
        elif flag == '51' :
            start_index = 3558
            end_index = 5012
            df1 = pd.read_csv('test_20240612_153851_dvl.csv')
            df2 = pd.read_csv('test_20240612_153851.csv')
    else:
        raise ValueError("Invalid mode")
    time =np.arange(N2)[start_index:end_index]
    predicted_v0 = predicted_v[start_index:end_index, 0]
    predicted_v1 = predicted_v[start_index:end_index, 1]
    DVL_V0 = DVL_V[start_index:end_index, 0]
    DVL_V1 = DVL_V[start_index:end_index, 1]
    GPS_V0 = GPS_V[start_index:end_index, 0]
    GPS_V1 = GPS_V[start_index:end_index, 1]   
    time_step = 1 / 5
    angles = df1.iloc[:, [0, 1, 2]].values
    velocities1 = df1.iloc[:, [4, 3, 5]].values
    rot_matrices = Rot.from_euler('xyz', angles).as_matrix()
    c_array1 = np.einsum('ijk,ik->ij', rot_matrices, velocities1)
    df1[['ve_1', 'vn_1']] = c_array1[:, :2]
    extracted_data = df1
    x1 =integrate_velocity(extracted_data['ve_1'], time_step)
    y1 =integrate_velocity(extracted_data['vn_1'], time_step)
    df1['ve_2'] = predicted_v0
    df1['vn_2'] = predicted_v1
    velocities2 = df1.iloc[:, [9, 8, 5]].values
    c_array1 = np.einsum('ijk,ik->ij', rot_matrices, velocities2)
    df1[['ve_2', 'vn_2']] = c_array1[:, :2]
    x2 =integrate_velocity(extracted_data['ve_2'], time_step)
    y2 =integrate_velocity(extracted_data['vn_2'], time_step) 
    df1['ve_3'] = GPS_V0
    df1['vn_3'] = GPS_V1
    velocities2 = df1.iloc[:, [11, 10, 5]].values
    c_array1 = np.einsum('ijk,ik->ij', rot_matrices, velocities2)
    df1[['ve_3', 'vn_3']] = c_array1[:, :2]
    x3 =integrate_velocity(extracted_data['ve_3'], time_step)
    y3 =integrate_velocity(extracted_data['vn_3'], time_step) 
    time_interval =  1 / 100
    velocities1 = [df2['4'][0]]
    velocities2 = [df2['5'][0]]
    a = df2.iloc[:, [5, 6, 7]].values
    angles = df2.iloc[:, [0, 1, 2]].values
    rot_matrices = Rot.from_euler('xyz', angles).as_matrix()
    c_array1 = np.einsum('ijk,ik->ij',rot_matrices,a)
    df2[['ax_1', 'ay_1']] = c_array1[:, :2]
    time_step = 1 / 100    
    for i in range(1, len(df2)):
        avg_acceleration =df2['ax_1'][i-1] 
        velocity_change = avg_acceleration * time_interval
        new_velocity = velocities1[-1] + velocity_change
        velocities1.append(new_velocity)
    df2['a_vx'] = velocities1
    for i in range(1, len(df2)):
        avg_acceleration =df2['ay_1'][i-1]
        velocity_change = avg_acceleration * time_interval
        new_velocity = velocities2[-1] + velocity_change
        velocities2.append(new_velocity)
    df2['a_vy'] = velocities2
    a_x =integrate_velocity(df2['a_vx'], time_step)
    a_y =integrate_velocity(df2['a_vy'], time_step)
    x4 = a_x.iloc[::20]
    y4 = a_y.iloc[::20]
    #######计算ATE和RTE##########
    # ate_ls = calculate_ate(x3, y3, x1, y1)
    # ate_pr = calculate_ate(x3,y3,x2,y2)
    # ate_ins = calculate_ate(x3,y3,x4,y4)
    # rte_ls = calculate_rte(x3,y3,x1,y1)
    # rte_pr = calculate_rte(x3,y3,x2,y2)
    # rte_ins = calculate_rte(x3,y3,x4,y4)

    # print(ate_ls)
    # print(ate_pr)
    # print(ate_ins)
    # print(rte_ls)
    # print(rte_pr)
    # print(rte_ins)

    plt.figure(1)
    plt.plot(x1, y1, label='origin', color='#EE0000', linestyle='solid', marker='*', markersize=4,  markevery=int(0.06*len(time)))
    plt.plot(x2, y2, label='predict',color='#8A2BE2', linestyle='solid', marker='d', markersize=4,  markevery=int(0.06*len(time)))
    plt.plot(x3, y3, label='gps',color='#43CD80', linestyle='solid', marker='^', markersize=4,  markevery=int(0.06*len(time)))
    plt.plot(x4, y4, label='SINS',color='#00BFFF', linestyle='solid', marker='^', markersize=4,  markevery=int(0.06*len(time)))

    plt.xlabel('East(m)')
    plt.ylabel('North(m)')
    plt.tight_layout()
    plt.legend(loc='center', bbox_to_anchor=(0.9,0.88), borderaxespad=0,ncol=1)
    # plt.savefig(f'{flag}_Trajectory.pdf')

    plt.figure(2)
    plt.subplot(2, 1, 1)  
    plt.plot(time, DVL_V0, label='origin', color='#EE0000', linestyle='solid')
    plt.plot(time, predicted_v0, label='predict',color='#8A2BE2', linestyle='solid')
    plt.plot(time, GPS_V0, label='gps', color='#43CD80', linestyle='solid')
    plt.title('Velocity')

    # plt.xlabel('Time')
    plt.ylabel('VL(m/s)')

    plt.xticks([])
    plt.tight_layout()


    plt.subplot(2, 1, 2)  
    plt.plot(time, DVL_V1, label='origin', color='#EE0000', linestyle='solid')
    plt.plot(time, predicted_v1, label='predict',color='#8A2BE2', linestyle='solid')
    plt.plot(time, GPS_V1, label='gps', color='#43CD80', linestyle='solid')

    plt.xlabel('Time(s)')
    plt.ylabel('VF(m/s)')
    plt.legend(loc='center', bbox_to_anchor=(0.5,1.1), borderaxespad=0,ncol=3)
    # plt.legend()
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(bottom=0.1)

    # plt.tight_layout()
    # plt.savefig(f'{flag}_Vectory.pdf')


    plt.show()