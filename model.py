import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset,random_split
import matplotlib.pyplot as plt

#处理： min-max归一化 ＋ dropout

#神经网络
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(784,256),nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,128),nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,10)
        )


    def forward(self,x):
        return self.net(x)

#数据处理
class MnistCsv(Dataset):
    def __init__(self,path):
        super().__init__()
        data=np.loadtxt(path,delimiter=',',skiprows=1)
        labels=data[:,0]
        images=data[:,1:]/255.0
        self.x=torch.tensor(images,dtype=torch.float32)
        self.y=torch.tensor(labels,dtype=torch.long)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    

data=MnistCsv("./mnist/mnist_train.csv")

train_size=int(len(data)*0.8)
test_size=len(data)-train_size

train_data,test_data=random_split(data,[train_size,test_size])

train_loader=DataLoader(train_data,batch_size=128,shuffle=True)
test_loader=DataLoader(test_data,batch_size=128,shuffle=False)

#预处理
model=MLP().to("cuda:0")
epochs=15                
optimizer=torch.optim.Adam(model.parameters(),lr=0.007)
loss_fn=nn.CrossEntropyLoss()
loss_history=[]

#训练
model.train()
for epoch in range(epochs):
    for x,y in train_loader:
        x,y=x.to("cuda:0"),y.to("cuda:0")
        y_hat=model(x)
        loss=loss_fn(y_hat,y)
        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

torch.save(model.state_dict(),"./mnist/model.pth")

#画图
figure=plt.figure()
plt.plot(range(len(loss_history)),loss_history)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

#评估
model.eval()
correct=0
total=0

with torch.no_grad():
    for x,y in test_loader:
        x,y=x.to("cuda:0"),y.to("cuda:0")
        y_hat=model(x)
        pred=torch.argmax(y_hat,dim=1)
        correct+=(pred==y).sum().item()
        total+=y.size(0)

accurancy=(correct/total)*100
print(f"准确率为{accurancy:.2f}%")