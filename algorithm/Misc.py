import numpy as np
from numpy import ndarray as narr
import matplotlib.pyplot as plt
from typing import Callable,Union
import itertools

def generate_spuer_cube_boundary(d:int,n=8):

    xi=np.linspace(0.05,1,n)
    permutations = list(itertools.product([0, 1], repeat=d-1))
    permutations =np.array(permutations).astype(np.float32)
    dd=permutations.shape[0]
    batch_size=n*d*dd
    points=np.zeros([batch_size,d])
    values=np.zeros(batch_size)
    for i in range(d):
        for j in range(n):
            points[(i*n+j)*dd:(i*n+j+1)*dd,:]=np.insert(permutations,i,xi[j],axis=1)
            values[(i*n+j+1)*dd-1]=xi[j]
    return points,values
   
def generate_grid(d:int, density:float):
    """
    生成在 (0,1)^d 内的等距格点

    参数:
    dimensions: int - 维数
    density: float - 每个维度的密度

    返回:
    grid_points: np.ndarray - 生成的等距格点
    """
    # 创建一个在 (0,1) 区间内的等距点
    points = np.arange(0, 1 + density, density)

    grid = np.meshgrid(*[points] * d, indexing='ij')
    grid_points = np.stack(grid, axis=-1).reshape(-1, d)

    return grid_points