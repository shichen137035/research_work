import numpy as np
from numpy import ndarray as narr
import matplotlib.pyplot as plt
from typing import Callable,Union
from basic import statistic_func


def generate_n_unique_colors(n:int)->list:
    """
    生成 n 个不重复的颜色，使用 matplotlib colormap。
    """
    cmap = plt.cm.get_cmap('tab10', n)  # 使用 matplotlib 中的 'tab10' colormap
    colors = [cmap(i) for i in range(n)]  # 获取 n 个颜色
    return colors

class draw_lib():
    def __init__(self, title_fontsize=28, tick_label_fontsize=20, tick_fontsize=16, frame_width:int =2, frame_color: str ='blue',
                 loc='best', fontsize=20, legend_title=None, legend_shadow=False, 
                 fancybox=True, frameon=True, edge_color='blue', bbox_to_anchor=None) -> None:
        self.frame_width= frame_width
        self.frame_color= frame_color
        self.legend_loc=loc
        self.legend_fontsize=fontsize
        self.legend_title=legend_title
        self.legend_shadow=legend_shadow
        self.legend_fancybox=fancybox
        self.legend_drameon=frameon
        self.legend_edge_color=edge_color
        self.legend_bbox_to_anchor=bbox_to_anchor
        self.title_fontsize=title_fontsize
        self.tick_fontsize=tick_fontsize
        self.tick_label_fontsize=tick_label_fontsize
        
    def subplot_4_setting(self) -> None:
        self.title_fontsize = 14  # 标题字体缩小
        self.tick_label_fontsize = 10  # 坐标轴刻度标签字体
        self.tick_fontsize = 8  # 坐标轴刻度字体
        self.frame_width = 1  # 细边框
        self.frame_color = "black"  # 边框颜色设为黑色，提高可读性

        self.legend_loc = "upper right"  # 图例位置，避免遮挡数据
        self.legend_fontsize = 10  # 图例字体
        self.legend_title = None  # 取消图例标题，节省空间
        self.legend_shadow = False  # 取消阴影，减少视觉干扰
        self.legend_fancybox = True  # 使用圆角框，提高美观度
        self.legend_frameon = False  # 取消图例边框，节省空间
        self.legend_edge_color = "black"  # 图例边框颜色
        self.legend_bbox_to_anchor = (1, 1)  # 调整图例位置
        
    def set_frame(self):
        ax = plt.gca()#获取边框
        ax.spines['top'].set_color(self.frame_color)  
        ax.spines['right'].set_color(self.frame_color)  
        ax.spines['bottom'].set_color(self.frame_color)
        ax.spines['left'].set_color(self.frame_color)
        ax.spines['bottom'].set_linewidth(self.frame_width)
        ax.spines['left'].set_linewidth(self.frame_width)
        ax.spines['top'].set_linewidth(self.frame_width)
        ax.spines['right'].set_linewidth(self.frame_width)
    
    def set_legend(self,ax=None):
        if ax==None:
            plt.legend(loc=self.legend_loc, fontsize=self.legend_fontsize, 
                   title=self.legend_title, shadow=self.legend_shadow, fancybox=self.legend_fancybox,
                   frameon=self.legend_drameon, edgecolor=self.legend_edge_color, bbox_to_anchor=self.legend_bbox_to_anchor)
        else:
            ax.legend(loc=self.legend_loc, fontsize=self.legend_fontsize, 
                   title=self.legend_title, shadow=self.legend_shadow, fancybox=self.legend_fancybox,
                   frameon=self.legend_drameon, edgecolor=self.legend_edge_color, bbox_to_anchor=self.legend_bbox_to_anchor)
            
    def set_inner_look(self,title:str ='Bar',xlabel='$x$',ylabel='$y$',x_color='black'
                ,y_color='black',xs_color='black',ys_color='black',grid_color='grey',alpha=0.1):
        plt.rc('font',family='Times New Roman')
        plt.title(title,fontsize=self.title_fontsize)
        plt.xlabel(xlabel,fontsize=self.tick_label_fontsize,color=x_color)
        plt.ylabel(ylabel,fontsize=self.tick_label_fontsize,color=y_color)
        plt.xticks(fontsize=self.tick_fontsize,color=xs_color)
        plt.yticks(fontsize=self.tick_fontsize,color=ys_color)
        if grid_color!=None and alpha!=None:
            plt.grid(color = grid_color,alpha=alpha)
    
    def set_3d_frame(self,ax):
        ax.w_xaxis.line.set_color(self.frame_color)
        ax.w_yaxis.line.set_color(self.frame_color)
        ax.w_zaxis.line.set_color(self.frame_color)
        ax.w_xaxis.line.set_linewidth(self.frame_width)
        ax.w_yaxis.line.set_linewidth(self.frame_width)
        ax.w_zaxis.line.set_linewidth(self.frame_width)
        
    def set_3d_inner_look(self,ax,title:str ='Bar',xlabel='$x$',ylabel='$y$', zlabel='$z$',x_color='black'
                ,y_color='black',z_color='black',xs_color='black',ys_color='black',zs_color='black',grid_color='grey',alpha=0.1):
        plt.rc('font', family='Times New Roman')
        ax.set_title(title, fontsize=28)
        ax.set_xlabel(xlabel, fontsize=20, color=x_color)
        ax.set_ylabel(ylabel, fontsize=20, color=y_color)
        ax.set_zlabel(zlabel, fontsize=20, color=z_color)
        
        ax.tick_params(axis='x', which='major', labelsize=16, color=xs_color)
        ax.tick_params(axis='y', which='major', labelsize=16, color=ys_color)
        ax.tick_params(axis='z', which='major', labelsize=16, color=zs_color)
        
        ax.grid(color=grid_color, alpha=alpha)
        
    def plot_hist(self,x:narr,title:str ='Hist',xlabel='$x$',ylabel='$y$',color='r',lw=1,label=None,x_color='black'
                ,y_color='black',xs_color='black',ys_color='black',grid_color='grey',alpha=0.1):
        plt.figure()
        self.set_frame()
        self.set_inner_look(title,xlabel,ylabel,x_color,y_color,xs_color,ys_color,grid_color,alpha)
        plt.hist(x,color=color,lw=lw,label=label)
        if label!=None:
            self.set_legend()  
    
    def plot_bar(self,x:narr,y:narr, bottom:narr =0,title:str ='Bar',xlabel='$x$',ylabel='$y$',color='r',lw=1,label=None,x_color='black'
                ,y_color='black',xs_color='black',ys_color='black',grid_color='grey',alpha=0.1):
        self.set_frame()
        self.set_inner_look(title,xlabel,ylabel,x_color,y_color,xs_color,ys_color,grid_color,alpha)
        plt.bar(x,y,bottom=bottom,color=color,lw=lw,label=label)
        if label!=None:
            self.set_legend()
    
    def plot_curve(self,x:narr,y:Union[narr|None] =None,title:str= r'$x$-$y$',xlabel='$x$',ylabel='$y$',color='r',lw=3,label=None,x_color='black'
                ,y_color='black',xs_color='black',ys_color='black',grid_color='grey',alpha=0.1):
        self.set_frame()
        self.set_inner_look(title,xlabel,ylabel,x_color,y_color,xs_color,ys_color,grid_color,alpha)
        if y is None:
            y=x
            x=np.linspace(0,y.shape[0]-1,y.shape[0])
        plt.xlim(x.min(),x.max())
        plt.ylim(y.min(),y.max()+0.05*y.max())
        plt.grid(color = grid_color,alpha=alpha)
        plt.plot(x,y,linestyle='-',color=color,lw=lw,label=label)
        if label!=None:
            self.set_legend()
    def plot_multi_curve(self,x:narr,*args,title:str ='multi curve',xlabel='$x$',ylabel='$y$',color:list =None,lw=3,label:Union[narr|list] =None,x_color='black'
                ,y_color='black',xs_color='black',ys_color='black',grid_color='grey',alpha=0.1):
        self.set_frame()
        self.set_inner_look(title,xlabel,ylabel,x_color,y_color,xs_color,ys_color,grid_color,alpha)
        plt.xlim(x.min(),x.max())
        plt.grid(color = grid_color,alpha=0.1)
        k=0
        if label==None:
            label=np.arange(0,len(args),1)
        if color==None:
            color=generate_n_unique_colors(len(args))
            
        for y in args:
            plt.plot(x,y,linestyle='-',color=color[k],lw=lw,label=label[k])
            k+=1
        if label!=None:
            self.set_legend()
    
    def scatter_2d(self,x1:narr,x2:narr,title:str,color='blue',xlabel='$x$',ylabel='$y$',x_color='black'
        ,y_color='black',label='scatter',xs_color='black',ys_color='black'):
        self.set_frame()
        self.set_inner_look(title,xlabel,ylabel,x_color,y_color,xs_color,ys_color,None,None)
        plt.scatter(x1,x2,label=label,color=color)
        if label!=None:
            self.set_legend()
    
    def scatter_3d(self, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, title: str ='Scatter of $x$', color='blue', xlabel='$x$', ylabel='$y$', zlabel='$z$', 
                x_color='black', y_color='black', z_color='black', label='scatter', xs_color='black', ys_color='black', zs_color='black', 
                grid_color='grey',alpha=0.1):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置图框加粗
        self.set_3d_frame(ax)
        self.set_3d_inner_look(ax,title,xlabel,ylabel,zlabel,x_color,y_color,z_color,xs_color,ys_color,zs_color,grid_color,alpha)
        ax.scatter(x1, x2, x3, label=label, color=color)
        if label!=None:
            self.set_legend()
        
        
        
def draw_loss(train_loss:narr,test_loss:narr,title='Loss in training',xlabel=r'$\text{epoch}$',ylabel=r'$\text{loss}$',color=['orange','skyblue'],lw=3,label=['train loss','test loss'],x_color='navy'
                ,y_color='red',xs_color='black',ys_color='black',grid_color='grey',alpha=0):
    ddd=draw_lib(title_fontsize=32,tick_fontsize=20,tick_label_fontsize=24)
    x=np.arange(1,train_loss.shape[0]+1,1)
    ddd.plot_multi_curve(x,train_loss,test_loss,title=title,xlabel=xlabel,ylabel=ylabel,color=color,lw=lw,label=label,x_color=x_color
                ,y_color=y_color,xs_color=xs_color,ys_color=ys_color,grid_color=grid_color,alpha=alpha)
    
def draw_distribution(data_set:narr,title='$F(x)$',xlabel=r'$x$',ylabel=r'$y$',color='r',lw=3,label='CDF',x_color='navy'
                ,y_color='red',xs_color='black',ys_color='black',grid_color='grey',alpha=0):
    ddd=draw_lib(title_fontsize=32,tick_fontsize=20,tick_label_fontsize=24)
    st=data_set.min()
    ed=data_set.max()
    F=statistic_func.empirical_distribution_function(data_set)
    w=np.linspace(st,ed,200)
    y=F(w)
    ddd.plot_curve(w,y,title=title,xlabel=xlabel,ylabel=ylabel,color=color,lw=lw,label=label,x_color=x_color
                ,y_color=y_color,xs_color=xs_color,ys_color=ys_color,grid_color=grid_color,alpha=alpha)

def draw_multi_distribution(*args:Union[list|narr],title='$F(x)$',xlabel=r'$x$',ylabel=r'$y$',color=None,lw=3,label=None,x_color='navy'
                ,y_color='red',xs_color='black',ys_color='black',grid_color='grey',alpha=0):
    ddd=draw_lib(title_fontsize=32,tick_fontsize=20,tick_label_fontsize=24)
    st=np.inf
    ed=-np.inf
    for u in args:
        st=min(u.min(),st)
        ed=max(u.max(),ed)
    w=np.linspace(st,ed,200)
    y=[]
    for i in range(len(args)):
        F=statistic_func.empirical_CDF(args[i])
        y0=F(w)
        y.append(y0)
    ddd.plot_multi_curve(w,*y,title=title,xlabel=xlabel,ylabel=ylabel,color=color,lw=lw,label=label,x_color=x_color
                ,y_color=y_color,xs_color=xs_color,ys_color=ys_color,grid_color=grid_color,alpha=alpha)
    
    
def plot_subplots(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray,
                   y1: np.ndarray, y2: np.ndarray, y3: np.ndarray, y4: np.ndarray,
                   titles: list = ["Plot 1", "Plot 2", "Plot 3", "Plot 4"],
                   xlabels: list = ["x1", "x2", "x3", "x4"],
                   ylabels: list = ["y1", "y2", "y3", "y4"],
                   colors: list = ['r', 'g', 'b', 'm'],
                   lw: float = 2):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    
    datasets = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    for i, ax in enumerate(axes):
        x, y = datasets[i]
        ax.plot(x, y, linestyle='-', color=colors[i], lw=lw)
        ax.set_title(titles[i])
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
        ax.grid(color='grey', alpha=0.3)
    
    plt.tight_layout()
    
