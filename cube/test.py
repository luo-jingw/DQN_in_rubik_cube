import cmd
from cube import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()
        self.linear1=nn.Linear(54,128)
        self.linear2=nn.Linear(128,256)
        self.linear3=nn.Linear(256,512)
        self.linear4=nn.Linear(512,512)
        self.linear5=nn.Linear(512,256)
        self.linear6=nn.Linear(256,13)       
        self.activation=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
                

        
    
    def forward(self, x):
        x=x.to(torch.float32)
        x=self.linear1(x)
        x=self.activation(x)
        x=self.linear2(x)
        x=self.activation(x)
        x=self.linear3(x)
        x=self.activation(x)
        x=self.linear4(x)
        x=self.activation(x)
        x=self.linear5(x)
        x=self.sigmoid(x)
        x=self.linear6(x)
        return x
    
    
    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.bias.data)   
    
# 模型加载
path='deep_q_learning_model.pkl'
try:
    net=torch.load(path)
    print("load model:",path)
except:
    print("wrong path")
    exit()
    
    
c0=cube()
c=cp.deepcopy(c0)
c.rubik(20)
c.display()
k=0
while c.losscube(c0)!=0 and k<30:
    test=torch.tensor(np.reshape(c.arr,(1,54)))
    control=net(test)
    s=torch.argmax(control)
    c.control(s.to(int))
    c.display()
    #print(control)
    print(c.controllist[s.to(int)],end=' ')
    k+=1

