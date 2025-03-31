import numpy as np
from numpy import ndarray as narr
import matplotlib.pyplot as plt
from typing import Callable,Union
from data_plot import *
import math
import scipy.stats as stats

def F(x:narr,theta:float):
    return x**theta

x=np.linspace(0,1,100)
y1=F(x,0.5)
y2=F(x,1)
y3=F(x,2)
y4=F(x,3)
ddd=draw_lib()
plt.figure()
plt.subplot(2,2,1)
ddd.plot_curve(x,y1)
plt.subplot(2,2,2)
ddd.plot_curve(x,y2)
plt.subplot(2,2,3)
ddd.plot_curve(x,y3)
plt.subplot(2,2,1)
ddd.plot_curve(x,y4)
plt.show()