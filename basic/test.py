import numpy as np
from numpy import ndarray as narr
import matplotlib.pyplot as plt
from typing import Callable,Union
from statistic_func import *

x1=np.random.uniform(0,1,size=128)
x2=np.random.uniform(0,1,size=128)
y=x1-x2
F=empirical_distribution_function(y)
w=np.arange(-1,1,0.02)
dis=F(w)
plt.plot(w,dis)
plt.show()
