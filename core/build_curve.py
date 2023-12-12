import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cal_factorial(num):
    product=1
    if num!=0:
        for idx in range(1,num+1):
            product = product * idx
    return product

def generate_bezier_param(degree,norm_time):
    whole_params=[]

    for i in range(1,degree+1):
        param1= cal_factorial(degree)/(cal_factorial(i)*cal_factorial(degree-i))
        param2= (1-norm_time)**(degree-i)
        param3= norm_time**i
        param = param1*param2*param3
        whole_params.append(param)

    bezier_matrix=torch.from_numpy(np.array(whole_params).transpose(1,2,0)) # degree*degree
    return bezier_matrix

def b_spline_basis(i, k, u, nodeVector):

    if k==0:
        result = np.where(np.logical_and(u >= nodeVector[i], u < nodeVector[i+1]),1,0)
    else:
        length1 = nodeVector[i + k] - nodeVector[i]
        length2 = nodeVector[i + k + 1] - nodeVector[i + 1]
        if length1 == 0:  #  0/0 = 0
            alpha = 0
        else:
            alpha = (u - nodeVector[i]) / length1
        if length2 == 0:
            beta = 0
        else:
            beta = (nodeVector[i + k + 1] - u) / length2
        result = alpha * b_spline_basis(i, k - 1, u, nodeVector) + beta * b_spline_basis(i + 1, k - 1, u, nodeVector)
        
    return result
    
def b_spline(control_num,degree,predict_moments, nodeVector):
    whole_params = []
    for i in range(control_num):
        single_control_weight = b_spline_basis(i, degree, predict_moments, nodeVector)
        whole_params.append(single_control_weight)    
    whole_params = np.array(whole_params)
    whole_params[-1,:,-1] = 1
    B_spline = torch.from_numpy(whole_params.transpose(1,2,0))
    return B_spline

def resize_flow(flow, new_shape):
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = F.interpolate(flow, (new_h, new_w),mode='bilinear', align_corners=True)
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow

if __name__ == '__main__':
    #bezier_matrix=generate_bezier_param(3,[0.1,0.2,0.3,0.4,0.5])
    u = np.array([[0,1/36,3/36,4/36,12/36,24/36,1]])
    nodeVector = np.array([0,0,0,0,1/3,2/3,1,1,1,1])
    B_spline = b_spline(6,3,u,nodeVector)
