import cmd
from cube import *
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()
        self.linear0=nn.Linear(54,9)
        self.linear1=nn.Linear(54,256)
        self.linear2=nn.Linear(256,128)
        self.linear3=nn.Linear(128,64)
        self.linear4=nn.Linear(64,13)       
        self.activation=nn.ReLU()
                

        
    
    def forward(self, x):
        x=x.to(torch.float32)
        x=x.to(device)
        x1=self.linear0(x[0:54])
        x2=self.linear0(x[54:108])
        x3=self.linear0(x[108:162])
        x4=self.linear0(x[162:216])
        x5=self.linear0(x[216:270])
        x6=self.linear0(x[270:324])
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
path='mydqn_model.pkl'
try:
    net=torch.load(path)
    net=net.to(device)
    net_=cp.deepcopy(net)
    print("continue training\nModel pkl Path:",path)
except:
    net=mynet().to(device)
    net.initialize()
    net_=cp.deepcopy(net)
    print("start training\nModel pkl Path:",path)
    
    
c0=cube()
loss_f=nn.MSELoss().to(device)
optim=torch.optim.RMSprop(net.parameters(),lr=0.0001)

gamma=0.9
#(st,at,rt,st+1)

Dsize=50000
batchsize=50
epsilon=0.5
D=deque()
def update(N):
    c=cp.deepcopy(c0)
    rubikstep=np.random.randint(N)
    c.rubik(rubikstep)
    for i in range(rubikstep):
        terminal=0
        st=torch.tensor(c.arr.flatten().astype(np.int64))-1
        st=F.one_hot(st).flatten()
        if np.random.rand()<epsilon:
            action=np.random.randint(13)
            action=torch.tensor(action,dtype=torch.int64)
        else:
            action=torch.argmax(net(st.to(device)))
        at=F.one_hot(action,13)
        rt=c.reward(c0)
        if c.losscube(c0)==0:
            terminal=1
        gamma=1-torch.tensor(c.losscube(c0)/48.)
        c.control(action.item())
        st1=torch.tensor(c.arr.flatten().astype(np.int64))-1
        st1=F.one_hot(st1).flatten()
        if len(D)==Dsize:
            D.popleft()
        D.append((st,at,rt,st1,terminal,gamma))
        if terminal==1:
            break
        


def trainnet():
    epochs=500000
    observe=100
    global path
    global net_
    global gamma
    for i in range(epochs):
        N=np.random.randint(1,40)
        if (i+1)%500==0 and i>1:
            # 模型保存
            torch.save(net,path)
            
        if i<observe:
            update(N)
            print("\roberving",i+1,end=" ")
        
        elif i==observe:
            print("\nstart training")
        else:
            epsilon=max(0.0001,0.5-0.0001*i)
            minibatch = random.sample(D, batchsize)
            # get the batch variables
            s_j_batch = torch.stack([d[0] for d in minibatch]).to(device)
            a_batch = torch.stack([d[1] for d in minibatch]).to(device)
            r_batch = torch.tensor([d[2] for d in minibatch], dtype=torch.float32, device=device)
            s_j1_batch = torch.stack([d[3] for d in minibatch]).to(device)
            terminal_batch = torch.tensor([d[4] for d in minibatch], dtype=torch.float32, device=device)
            gamma_batch = torch.stack([d[5] for d in minibatch]).to(device)

            if i % 10 == 0:
                net_ = cp.deepcopy(net)

            # current Q values
            q_values = net(s_j_batch)
            q_values = torch.sum(q_values * a_batch, dim=1)

            with torch.no_grad():
                q_next = net_(s_j1_batch).max(1)[0]

            targets = r_batch + (1 - terminal_batch) * gamma_batch * q_next

            loss = loss_f(q_values, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

            print(
                "epoch/epochs", i, '/', epochs,
                f"| Q:{q_values.mean().item():>7.2f}",
                f"| gamma:{gamma_batch.mean().item():>4.3f}",
                f"| reward:{r_batch.mean().item():>5.1f}",
                f"| loss:{loss.item():>10.3f}",
                f"| epsilon:{epsilon:>5.4f}"
            )
            update(N)
            
                

if __name__=='__main__':
    print("is cuda available:",torch.cuda.is_available())
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
