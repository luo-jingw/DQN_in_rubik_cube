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
        self.linear01=nn.Linear(54,9)
        self.linear02=nn.Linear(54,9)
        self.linear03=nn.Linear(54,9)
        self.linear04=nn.Linear(54,9)
        self.linear05=nn.Linear(54,9)
        self.linear06=nn.Linear(54,9)
        self.linear1=nn.Linear(54,256)
        self.linear2=nn.Linear(256,128)
        self.linear3=nn.Linear(128,64)
        self.linear4=nn.Linear(64,13)       
        self.activation=nn.ReLU()
                

        
    
    def forward(self, x):
        x=x.to(torch.float32)
        x1=self.linear01(x[0:54])
        x2=self.linear02(x[54:108])
        x3=self.linear03(x[108:162])
        x4=self.linear04(x[162:216])
        x5=self.linear05(x[216:270])
        x6=self.linear06(x[270:324])
        x=torch.cat((x1,x2,x3,x4,x5,x6))
        x=self.linear1(x)
        x=self.activation(x)
        x=self.linear2(x)
        x=self.activation(x)
        x=self.linear3(x)
        x=self.activation(x)
        x=self.linear4(x)
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
batchsize=50
epsilon=0.5
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
    epochs=500000
    observe=800
    global path
    for i in range(epochs):
        if i<500:
            N=3
        if 500<=i<2000:
            N=10
        if i>2000:
            N=30
        if (i+1)%500==0 and i>1:
            # 模型保存
            torch.save(net,path)
            
        if i<observe:
            update()
            print("\roberving",i+1,end=" ")
        
        elif i==observe:
            print("\nstart training")
        else:
            epsilon=max(0.0001,0.4-0.00001*i)
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
