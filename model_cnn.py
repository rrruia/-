import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os

# dropout + 数据增强 + 标准化 + BatchNorm归一化


device="cuda:0"

#CNN网络

class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(256*14*14,512),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(512,2)
        )

    def forward(self,x):
        return self.net(x)

#数据处理

train_transformer=transforms.Compose([
    transforms.RandomResizedCrop(224,(0.8,1),(0.9,1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

test_transformer=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

class CatDogDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []

        for file in os.listdir(root):
            if file.endswith((".jpg", ".png")):
                if "cat" in file.lower():
                    label = 0
                elif "dog" in file.lower():
                    label = 1
                else:
                    continue
                self.images.append((os.path.join(root, file), label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


train_data = CatDogDataset("./dog_vs_cat/train", transform=train_transformer)
test_data = CatDogDataset("./dog_vs_cat/test", transform=test_transformer)

train_loader=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True)
test_loader=DataLoader(test_data,batch_size=128,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)

#预处理

model=CatDogCNN().to(device)

epoches=30
optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
loss_fn=nn.CrossEntropyLoss()
loss_history=[]

#训练


if __name__ == '__main__':
    model.train()
    for epoch in range(epoches):
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            y_hat=model(x)
            loss=loss_fn(y_hat,y)
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

    torch.save(model.state_dict(),"./dog_vs_cat/model_cnn.pth")


    #画图

    fig=plt.figure()
    plt.plot(range(len(loss_history)),loss_history)
    plt.xlabel("times")
    plt.ylabel("losses")
    plt.show()

    #测试

    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device)
            output=model(x)
            pred=torch.argmax(output,dim=1)
            correct+=(pred==y).sum().item()
            total+=y.size(0)

    accuracy=(correct/total)*100
    print(f"准确率为{accuracy:.4f}%")