import numpy as np
from numpy import ndarray as narr
import matplotlib.pyplot as plt
from typing import Callable,Union
from basic.calculus import value_safety_check
from scipy import stats
from basic.common_func import uniform_cdf


def empirical_distribution_function(X:narr):
    """
    ## Parameters:
    X : Array of observed values of the random variable X.
    ## Returns:
    F : the empirical distribution
    ## Description:
    Compute the empirical distribution function (ECDF) of a one-dimensional random variable X.
    """
    def F(x:Union[narr,float])->Union[narr,float]:
        n = len(X)
        if isinstance(x,narr):
            y=np.zeros(x.shape)
            
            for i in range(x.shape[0]):
                count = np.sum(X <= x[i])
                y[i]=count/(n+1)
            
            return y
        elif isinstance(x,float):
            count=np.sum(X<=x)
            return count/(n+1)
    return F    

def empirical_CDF(X:narr):
    n=len(X)
    X=np.sort(X)
    def F(x:Union[narr,float])->Union[narr,float]:
        y=np.searchsorted(X,x,side='right')
        y=y/(n+1)
        return y
    return F  

def give_p_value(T:narr,T0:float):
    count=np.sum(T>=T0)
    return (1+count)/(T.shape[0]+1)

def all_statistics(empirical_F:Callable[[narr],narr], theoretical_F:Callable[[narr],narr],N=50,type='KS')->float:
    """
    计算 Kolmogorov-Smirnov 统计量 \tau^{KS}
    
    参数:
    empirical_F (function): 经验分布函数 \hat{F}(w)
    theoretical_F (function): 理论分布函数 F(w)
    w_values (numpy array): w 在 [0, 1] 区间的离散取值
    
    返回:
    float: Kolmogorov-Smirnov 统计量 \tau^{KS}
    """
    w_values=np.linspace(1e-15,1-1/N,N)
    differences = np.abs(empirical_F(w_values) - theoretical_F(w_values))
    # 计算每个 w 上的差值
    if type=='KS':
        # 计算 KS 统计量
        tau_ks = np.sqrt(N) * np.max(differences)
        return tau_ks
    elif type=='CwM':
        tau_cwm = N*np.sum(differences**2)
        return tau_cwm
    elif type=='AD':
        xx=theoretical_F(w_values)*(1-theoretical_F(w_values))
        xx=value_safety_check(xx)
        tau_ad = N*np.sum(differences**2/xx)
        return tau_ad
    elif type=='KSAD':
        xx=theoretical_F(w_values)*(1-theoretical_F(w_values))
        xx=value_safety_check(xx)
        tau_tsad= np.sqrt(N) *np.max(differences/xx)
        return tau_tsad
    else:
        print('not included type')
        raise ValueError

def combine_p_values(p_values):
    # Fisher合并方法
    chi_square_stat = -2 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    combined_p_value = stats.chi2.sf(chi_square_stat, df)
    return combined_p_value    
    
def give_T(X:narr,type='KS')->float:
    F0=empirical_distribution_function(X)
    T0=all_statistics(F0,uniform_cdf,type=type)
    return T0