import numpy as np
from numpy import ndarray as narr
import matplotlib.pyplot as plt
from typing import Callable,Union
import os

def uniform_cdf(w:narr)->narr:
    return w

def save_sample(X:narr,folder:str,name:str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 构建完整的文件路径
    file_path = os.path.join(folder, name)
    
    # 保存数组到文件
    np.save(file_path, X)

def load_sample(folder: str, name: str) -> narr:
    # 构建完整的文件路径
    name=name+'.npy'
    file_path = os.path.join(folder, name)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")
    
    # 加载数组并返回
    return np.load(file_path)