import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import time

np.random.seed(int(time.time()))


class cube():
    def __init__(self):
        self.arr=np.zeros(shape=(6,3,3))
        self.controllist=['F','B','R','L','U','D','f','b','r','l','u','d','o']
        s1=np.ones(shape=(3,3))
        s2=2*s1
        s3=3*s1
        s4=4*s1
        s5=5*s1
        s6=6*s1
        self.arr[0]=s1
        self.arr[1]=s2
        self.arr[2]=s3
        self.arr[3]=s4
        self.arr[4]=s5
        self.arr[5]=s6
        self.viewarr=np.zeros((9,12))
        
        
    
    def F(self):
        arr=cp.deepcopy(self.arr)
        arr[0]=np.rot90(arr[0],k=1)
        t=cp.deepcopy(arr[1][0,:])
        arr[1][0,:]=arr[4][0,:]
        arr[4][0,:]=arr[3][0,:]
        arr[3][0,:]=arr[2][0,:]
        arr[2][0,:]=t
        self.arr=cp.deepcopy(arr)
        
        
    def f(self):
        arr=cp.deepcopy(self.arr)
        arr[0]=np.rot90(arr[0],k=3)
        t=cp.deepcopy(arr[1][0,:])
        arr[1][0,:]=arr[2][0,:]
        arr[2][0,:]=arr[3][0,:]
        arr[3][0,:]=arr[4][0,:]
        arr[4][0,:]=t
        self.arr=cp.deepcopy(arr)
        
        
    def B(self):
        arr=cp.deepcopy(self.arr)
        arr[5]=np.rot90(arr[5],k=1)
        t=cp.deepcopy(arr[1][2,:])
        arr[1][2,:]=arr[2][2,:]
        arr[2][2,:]=arr[3][2,:]
        arr[3][2,:]=arr[4][2,:]
        arr[4][2,:]=t
        self.arr=cp.deepcopy(arr)
        
    
    def b(self):
        arr=cp.deepcopy(self.arr)
        arr[5]=np.rot90(arr[5],k=3)
        t=cp.deepcopy(arr[1][2,:])
        arr[1][2,:]=arr[4][2,:]
        arr[4][2,:]=arr[3][2,:]
        arr[3][2,:]=arr[2][2,:]
        arr[2][2,:]=t
        self.arr=cp.deepcopy(arr)
        
        
    def R(self):
        arr=cp.deepcopy(self.arr)
        arr[1]=np.rot90(arr[1],k=1)
        t=cp.deepcopy(arr[2][:,0])
        arr[2][:,0]=arr[5][:,0]
        arr[5][:,0]=np.flipud(arr[4][:,2])
        arr[4][:,2]=np.flipud(arr[0][:,0])
        arr[0][:,0]=t
        self.arr=cp.deepcopy(arr)
        
        
    def r(self):
        arr=cp.deepcopy(self.arr)
        arr[1]=np.rot90(arr[1],k=3)
        t=cp.deepcopy(arr[2][:,0])
        arr[2][:,0]=arr[0][:,0]
        arr[0][:,0]=np.flipud(arr[4][:,2])
        arr[4][:,2]=np.flipud(arr[5][:,0])
        arr[5][:,0]=t
        self.arr=cp.deepcopy(arr)
        
        
        
    def L(self):
        arr=cp.deepcopy(self.arr)
        arr[3]=np.rot90(arr[3],k=1)
        t=cp.deepcopy(arr[2][:,2])
        arr[2][:,2]=arr[0][:,2]
        arr[0][:,2]=np.flipud(arr[4][:,0])
        arr[4][:,0]=np.flipud(arr[5][:,2])
        arr[5][:,2]=t
        self.arr=cp.deepcopy(arr)
        
        
    def l(self):
        arr=cp.deepcopy(self.arr)
        arr[3]=np.rot90(arr[3],k=3)
        t=cp.deepcopy(arr[2][:,2])
        arr[2][:,2]=arr[5][:,2]
        arr[5][:,2]=np.flipud(arr[4][:,0])
        arr[4][:,0]=np.flipud(arr[0][:,2])
        arr[0][:,2]=t
        self.arr=cp.deepcopy(arr)
        
        
    def U(self):
        arr=cp.deepcopy(self.arr)
        arr[4]=np.rot90(arr[4],k=1)
        t=cp.deepcopy(arr[0][0,:])
        arr[0][0,:]=np.flipud(arr[1][:,0])
        arr[1][:,0]=arr[5][2,:]
        arr[5][2,:]=np.flipud(arr[3][:,2])
        arr[3][:,2]=t
        self.arr=cp.deepcopy(arr)
        
        
    def u(self):
        arr=cp.deepcopy(self.arr)
        arr[4]=np.rot90(arr[4],k=3)
        t=cp.deepcopy(arr[0][0,:])
        arr[0][0,:]=arr[3][:,2]
        arr[3][:,2]=np.flipud(arr[5][2,:])
        arr[5][2,:]=arr[1][:,0]
        arr[1][:,0]=np.flipud(t)
        self.arr=cp.deepcopy(arr)   
        
    
    def D(self):
        arr=cp.deepcopy(self.arr)
        arr[2]=np.rot90(arr[2],k=1)
        t=cp.deepcopy(arr[0][2,:])
        arr[0][2,:]=arr[3][:,0]
        arr[3][:,0]=np.flipud(arr[5][0,:])
        arr[5][0,:]=arr[1][:,2]
        arr[1][:,2]=np.flipud(t)
        self.arr=cp.deepcopy(arr)
        
        
    def d(self):
        arr=cp.deepcopy(self.arr)
        arr[2]=np.rot90(arr[2],k=3)
        t=cp.deepcopy(arr[0][2,:])
        arr[0][2,:]=np.flipud(arr[1][:,2])
        arr[1][:,2]=arr[5][0,:]
        arr[5][0,:]=np.flipud(arr[3][:,0])
        arr[3][:,0]=t
        self.arr=cp.deepcopy(arr)
        
        
    def o(self):
        pass
    
    
    
    
    def display(self):
        a=self.viewarray()
        #print('\n',a)
        ar=np.zeros(shape=(9,12,3))
        for i in range(9):
            for j in range(12):
                if a[i,j]==0:
                    ar[i,j]=[1,1,1]
                if a[i,j]==1:
                    ar[i,j]=[0,0,1]
                if a[i,j]==2:
                    ar[i,j]=[0,1,0]
                if a[i,j]==3:
                    ar[i,j]=[1,0,0]
                if a[i,j]==4:
                    ar[i,j]=[1,1,0]
                if a[i,j]==5:
                    ar[i,j]=[1,0,1]
                if a[i,j]==6:
                    ar[i,j]=[0,1,1]
        #print(a)
        plt.matshow(ar)
        plt.show()
        
    def viewarray(self):
        z=np.zeros(shape=(3,3))
        a1=np.hstack((z,self.arr[0],z,z))
        a2=np.hstack((self.arr[1],self.arr[2],self.arr[3],self.arr[4]))
        a3=np.hstack((z,self.arr[5],z,z))
        a=np.vstack((a1,a2,a3))
        return a
        
        
    def control(self,s):
        if s==0:
            self.F()
        if s==1:
            self.B()
        if s==2:
            self.R()
        if s==3:
            self.L()
        if s==4:
            self.U()
        if s==5:
            self.D()
        if s==6:
            self.f()
        if s==7:
            self.b()
        if s==8:
            self.r()
        if s==9:
            self.l()
        if s==10:
            self.u()
        if s==11:
            self.d()
        if s==12:
            self.o()
        self.viewarr=self.viewarray()
  
    
    def rubik(self,len):
        #随机打乱len步
        #输出打乱操作
        #print("rubik step:")
        for i in range(len):
            s=np.random.randint(12)
            self.control(s)
            #print(self.controllist[s],end=' ')
        #print('\n')
            
    def losscube(self,c0):
        lossc=0
        for i in range(6):
            for j in range(3):
                for k in range(3):
                    if self.arr[i,j,k]!=c0.arr[i,j,k]:
                        lossc+=1
        return lossc

    
    def reward(self,c0):
        if self.losscube(c0)==0:
            rt=1000
        else:
            rt=0
            for i in range(3):
                for j in range(3):
                    if self.arr[0][i,j]==c0.arr[0][i,j]:
                        rt=rt+1
                    if self.arr[5][i,j]==c0.arr[5][i,j]:
                        rt=rt+5
            for i in range(4):
                for j in range(3):
                    if self.arr[i+1][0,j]==c0.arr[i+1][0,j]:
                        rt=rt+1
                    if self.arr[i+1][1,j]==c0.arr[i+1][1,j]:
                        rt=rt+5
                    if self.arr[i+1][2,j]==c0.arr[i+1][2,j]:
                        rt=rt+10
        rt=rt-0.5*self.losscube(c0)
                    
                        
        return rt



