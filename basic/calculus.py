import numpy as np
from numpy import ndarray as narr
import matplotlib.pyplot as plt
from typing import Callable,Union


def generate_matrix_parallel(func:Callable[[narr],narr], u0:narr, delta_x=0.001,low_limit=0,high_limit=1)->narr:
    """
    生成围绕 u0 点的离散数值矩阵 D
    
    :param func: 多元函数
    :param u0: 求导的位置，可以是 (batch_size, n) 或 (n,)
    :param delta_x: 离散增量
    :param low_limit: 最低限值（可选）
    :param high_limit: 最高限值（可选）
    :return: 矩阵 D 和对应的函数值
    """
    # 检查输入形状，确定是否有批次维度
    if len(u0.shape) == 1:
        # 如果是单个点，没有批次维度，则增加一个虚拟批次维度
        u0 = u0[np.newaxis, :]
    
    batch_size, n = u0.shape  # 获取批大小和变量个数
    D = np.zeros((batch_size,) + (3,) * n)  # 初始化结果矩阵
    
    # 生成偏移量的索引，形状为 (n, 3, 3, ..., 3)
    indices = np.indices((3,) * n) - 1  # 索引值为 [-1, 0, 1]
    
    # 遍历每个点的索引并计算对应的函数值
    for idx in np.ndindex(indices.shape[1:]):
        offset = indices[(slice(None),) + idx]  # 偏移向量
        u = u0[:] + offset * delta_x  # 广播计算新位置
        # print(u)
        # u=value_safety_check(u)
        # 计算批量函数值并存入矩阵 D
        D[(slice(None),) + idx] = func(u)
    return D

def generate_matrix(func:Callable[[narr],narr], u0:narr, delta_x:float,low_limit=0,high_limit=1):
    """
    生成围绕u0点的离散数值矩阵D
    :param func: 多元函数
    :param u0: 求导的位置
    :param delta_x: 离散增量
    :return: 包含u0点及其前后离散点的矩阵和对应的函数值
    """
    n = len(u0)
    D = np.zeros((3,) * n)
    po =np.zeros((3,) * n+(n,))
    indices = np.indices((3,) * n) - 1

    for idx in np.ndindex(indices.shape[1:]):
        offset = indices[(slice(None),) + idx]
        u = u0 + offset * delta_x
        u=value_safety_check(u,low_limit=low_limit,high_limit=high_limit)
        po[idx]=u
        D[idx] = func(u)
    # D=func(u)

    return D,po

def central_difference(D:narr, po:narr):
    """
    计算离散数值矩阵的中心差分
    :param D: 离散数值矩阵
    :param delta_x: 离散增量
    :return: 中心差分值
    """
    n = len(D.shape)
    for dim in range(n):
        # 定义沿当前维度的切片
        slice_fwd = [slice(None)] * n
        slice_bwd = [slice(None)] * n
        slice_mid = [slice(None)] * n
        
        slice_fwd[dim] = slice(2, None)
        slice_bwd[dim] = slice(None, -2)
        slice_mid[dim] = slice(1,2)
        po_diff=(po[tuple(slice_fwd)] - po[tuple(slice_bwd)])
        dis=np.sum(po_diff**2,axis=-1)
        dis=np.sqrt(dis)
        # 计算沿当前维度的中心差分
        D = (D[tuple(slice_fwd)] - D[tuple(slice_bwd)]) / dis
        po=po[tuple(slice_mid)]
        po=po[...,1:]
        # print(D)
        # print(po)
    
    return D.item()  # 将最后的结果转换为标量

def central_difference_parallel(D:narr, delta_x=0.001)->narr:
    """
    计算离散数值矩阵的中心差分
    :param D: 离散数值矩阵
    :param delta_x: 离散增量
    :return: 中心差分值
    """
    # batch_size=D.shape[0]
    n = len(D.shape)-1
    for dim in range(1,n+1):
        # 定义沿当前维度的切片
        slice_fwd = [slice(None)] * (n+1)
        slice_bwd = [slice(None)] * (n+1)
        slice_mid = [slice(1, -1)] * (n+1)
        
        slice_fwd[dim] = slice(2, None)
        slice_bwd[dim] = slice(None, -2)
        
        D = (D[tuple(slice_fwd)] - D[tuple(slice_bwd)]) / (2 * delta_x)
    D=D.reshape(D.shape[0])
    return D

def calculate_centeral_difference(func:Callable[[narr],narr], u0:narr, delta_x=0.01):
    '''
    ## Parameters:
    func : the function that need to be differentiated
    u0 : data point 
    delta_x : differential delta
    ## Returns:
    Value : Generate centeral multi-partial
    ## Description:
    Use centeral differential to give differential
    '''
    D,po=generate_matrix(func,u0,delta_x)
    result=central_difference(D,po)
    # if len(result)==1:
    #     result=result.item()
    return result

def value_safety_check(*args,low_limit=0,high_limit=1)->list:
    '''
    ## Parameters:
    args : arrays that need to be checked
    low_limit : low_value boundary
    high_limit : high_value_limit
    ## Returns:
    result : values that pass safety check
    ## Descrption:
    a safety check that used to avoid situation like divide by 0 by adding a small number to it.
    '''
    result=[]
    for u in args:
        if isinstance(u,narr) or isinstance(u,float):
            u[u<=low_limit]=low_limit+1e-15
            u[u>=high_limit]=high_limit-1e-15
            result.append(u)
        else:
            print('wrong type!')
            raise TypeError
    if len(result)==1:
        return result[0]
    else:
        return result