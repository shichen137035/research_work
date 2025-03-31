import numpy as np
import torch
import torch.optim
import torch.nn as nn
from typing import Union,Callable
import time
from numpy import ndarray as narr
from sklearn.model_selection import train_test_split



def optimizer(model: nn.Module, lr=0.01, step_size=100, gama=0.5):
    '''
    Input: model is the network you use(transform e.g.),lr is learning rate,step_size is how many steps learning rate will decay,
    gama is how much it decays
    Output: opti is the adam optimizer(can be changed), scheduler is the rule of learning rate changing
    '''
    opti = torch.optim.AdamW(model.parameters(),
                            lr=lr,weight_decay=1e-5)  #optimizer,lr as the learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(opti,
                                                step_size=step_size,
                                                gamma=gama)
    #learning rate decays by gama for each time steps
    return opti, scheduler

def train_model(model: nn.Module,
          optimizer,
          sche,
          X: torch.tensor,
          target,
          loss_func:Callable[[torch.Tensor,torch.Tensor],float],
          epoch=100):
    loss_record = np.zeros(epoch)  # record the loss curve
    st = time.time()
    for i in range(epoch):
        prediction = model.forward(X)  # calculate prediction
        loss = loss_func(prediction, target)  # calculate loss
        loss_record[i] = loss
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        sche.step()  # update learning rate
        print('loss = {:.4f}; training {:.2f}% complete   '.format(
            loss, (i + 1) / epoch * 100),
              end="\r")
    end = time.time()
    cost = end - st  #time cost
    return loss_record, cost

def train_model_test_stop(model: nn.Module,
                optimizer,
                sche,
                X: torch.Tensor,
                target: torch.Tensor,
                loss_func: Callable[[torch.Tensor, torch.Tensor], float],
                test_ratio:float =0.33,
                lambda1: float =0.01,
                epoch: int =1000,  # 最大训练轮数
                patience: int =40,  # 早停耐心值
                save_path: str ="best_model.pth"):  # 最优模型保存路径
    
    # 划分训练集和测试集
    X_train, X_test, target_train, target_test = train_test_split(
    X, target,
    test_size=test_ratio,
    random_state=42,
    shuffle=True
)
    
    train_loss_record = []
    test_loss_record = []
    best_test_loss = float('inf')
    no_improve_count = 0
    if lambda1 is not  None:
        def L1_normalized_loss_function(prediction:torch.Tensor, target:torch.Tensor)->float:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            return loss_func(prediction,target)+lambda1*l1_norm
    else:
        def L1_normalized_loss_function(prediction:torch.Tensor, target:torch.Tensor)->float:
            return loss_func(prediction,target)
        
    st = time.time()

    # 训练循环
    for i in range(epoch):
        model.train()
        prediction = model(X_train)  # 训练集预测
        # loss = loss_func(prediction, target_train)  # 训练集损失
        loss = L1_normalized_loss_function(prediction, target_train)
        train_loss_record.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sche.step()
        
        # 计算测试集损失
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = loss_func(test_pred, target_test).item()
            test_loss_record.append(test_loss)
        
        print(f'Epoch [{i + 1}/{epoch}] - Train Loss: {loss.item():.6f}, Test Loss: {test_loss:.6f}')
        
        # 早停 & 最优模型保存
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            no_improve_count = 0
            
            # ✅ 保存最优模型参数
            torch.save(model.state_dict(), save_path)
            # print(f"🔹 New best model saved at epoch {i + 1} with test loss: {best_test_loss:.4f}")
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print(f"\n🛑 Early stopping at epoch {i + 1} with best test loss: {best_test_loss:.4f}")
            break

    # 训练完成后，加载最优模型参数
    print("\n🔄 Loading best model parameters...")
    model.load_state_dict(torch.load(save_path, weights_only=True))

    end = time.time()
    cost = end - st  # 计算训练时间
    train_loss_record=np.array(train_loss_record)
    test_loss_record=np.array(test_loss_record)

    return train_loss_record, test_loss_record, cost

def combine_loss_functions(*loss_funcs):
    """
    输入多个符合损失函数格式的函数 f(prediction, target)
    返回一个新的损失函数，该函数计算所有输入损失函数的和
    """
    def combined_loss(prediction, target):
        total_loss = sum(loss_func(prediction, target) for loss_func in loss_funcs)
        return total_loss
    return combined_loss


