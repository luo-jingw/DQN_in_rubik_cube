import cmd
from cube import *
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


 
class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()
        self.linear1=nn.Linear(324,128)
        self.linear2=nn.Linear(128,256)
        self.linear3=nn.Linear(256,512)
        self.linear4=nn.Linear(512,512)
        self.linear5=nn.Linear(512,256)
        self.linear6=nn.Linear(256,13)       
        self.activation=nn.ReLU()
                

        
    
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
        x=self.activation(x)
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
    print("continue training\nModel pkl Path:",path)
except:
    net=mynet()
    net.initialize()
    print("start training\nModel pkl Path:",path)
    
    
c0=cube()
loss_f=nn.MSELoss()
optim=torch.optim.RMSprop(net.parameters(),lr=0.0001)

gamma=0.9
#(st,at,rt,st+1)

N=50
Dsize=50000
batchsize=32
epsilon=0.9
D=deque()
def update():
    c=cp.deepcopy(c0)
    rubikstep=np.random.randint(N)
    c.rubik(rubikstep)
    for i in range(rubikstep):
        terminal=0
        st=torch.tensor(c.arr.flatten().astype(np.int64))-1
        st=F.one_hot(st).flatten()
        if np.random.rand()<epsilon:
            action=np.random.randint(12)
            action=torch.tensor(action,dtype=torch.int64)
        else:
            action=torch.argmax(net(st))
        at=F.one_hot(action,13)
        rt=c.reward(c0)
        c.control(action.item())
        st1=torch.tensor(c.arr.flatten().astype(np.int64))-1
        st1=F.one_hot(st1).flatten()
        if c.losscube(c0)==0:
            terminal=1
        if len(D)==Dsize:
            D.popleft()
        D.append((st,at,rt,st1,terminal))
        if terminal==1:
            continue
        


def trainnet():
    epochs=70000
    observe=100
    global path
    for i in range(epochs):
        if (i+1)%500==0 and i>1:
            # 模型保存
            path=path
            torch.save(net,path)
            
        if i<observe:
            update()
            print("\roberving",i+1,end=" ")
        
        elif i==observe:
            print("\nstart training")
        else:
            epsilon=max(0.0001,0.9-0.0001*i)
            minibatch = random.sample(D, batchsize)
            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            terminal_batch=[d[4] for d in minibatch]
            for j in range(batchsize):
                st=s_j_batch[j]
                at=a_batch[j]
                rt=torch.tensor(r_batch[j],dtype=torch.float32)
                st1=s_j1_batch[j]
                qt=net(st)
                qt1=net(st1)
                if terminal_batch[j]==0:
                    loss=loss_f(torch.sum(torch.multiply(qt,at)),rt+gamma*torch.max(qt1))
                else:
                    loss=loss_f(torch.sum(torch.multiply(qt,at)),rt)
                optim.zero_grad()
                loss.backward()
                optim.step()
                print("epoch/epochs",i,'/',epochs,\
                "| Q:%.3f\t"%(torch.sum(torch.multiply(qt,at)).cpu().detach().item()),\
                "| loss:%.6f\t"%(loss.item()),\
                "| action:%s"%(c0.controllist[torch.argmax(at).item()]),\
                "| reward:%s\t"%(rt.cpu().detach().item()),\
                "| epsilon:%.5f\t"%(epsilon))
            update()
            
                

if __name__=='__main__':
    trainnet()
    c0=cube()
    c=cp.deepcopy(c0)
    c.rubik(20)
    c.display()
    k=0
    while c.losscube(c0)!=0 and k<30:
        test=torch.tensor(c.arr.flatten().astype(np.int64))-1
        test=F.one_hot(test).flatten()
        control=net(test)
        s=torch.argmax(control)
        c.control(s.to(int))
        #print(control)
        print(c.controllist[s.to(int)],end=' ')
        k+=1
    c.display()
