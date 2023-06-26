# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from torch.autograd import Variable
from torchvision import transforms
import time
import os
import torch.backends.cudnn as cudnn


def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    '''
    数据增强
    # transforms.Compose(),将一系列的transforms有序组合,实现按照这些方法依次对图像操作
    # ToTensor()使图片数据转换为tensor张量,这个过程包含了归一化,图像数据从0~255压缩到0~1,这个函数必须在Normalize之前使用
    # 实现原理：针对不同类型进行处理,将各值除以255,最后通过torch.from_numpy将PIL Image或者 numpy.ndarray()针对具体类型转成torch.tensor()数据类型
    # Normalize()是归一化过程,ToTensor()的作用是将图像数据转换为(0,1)之间的张量,Normalize()则使用公式(x-mean)/std,将每个元素分布到(-1,1). 归一化后数据转为标准格式
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=transform_train)
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=transform_test)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=128, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=128, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

# 定义VGG模型
class Vgg(nn.Module):
    def __init__(self, num_classes=10):
        super(Vgg, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 16*16*64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8*8*128

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 4*4*256

            nn.Dropout(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2*2*512
        )
        self.classifier1 = nn.Linear(2048, 32)
        self.classifier2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

def start():
    # 开始训练
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    start_time = time.time()
    learning_rate = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 实例化vgg模型
    model = Vgg().to(device)
    # 优化器选用Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 损失函数使用交叉熵
    criterion = nn.CrossEntropyLoss()

    dataset_train, dataset_val, data_loader_train, data_loader_val = read_data()

    # 定义训练函数
    def train(epoch):
        model.train()
        train_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, labels) in enumerate(data_loader_train):
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            # 前向传播过程
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 反向传播过程
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()   # 计算训练误差，是每个批量训练误差的和
            # torch.max()返回两个值，第一个值是具体的value（用下划线_表示），第二个值是value所在的index（也就是predicted）
            _, predicted = torch.max(outputs.data, 1)   # 得到所有inputs的预测值
            total += labels.size(0)  # 总训练样本数
            correct += predicted.eq(labels.data).sum()   # 正确分类数

            # 打印训练迭代次数，批量下标，损失，正确率
            if batch_idx % 10 == 0:
                print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                      .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # 定义测试函数
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(data_loader_val):
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)     # 总的样本数
            correct += predicted.eq(labels.data).sum()    # 正确分类的数量
        # 打印测试误差和正确率
        print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
              .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        return test_loss

    for epoch in range(0, 30):  # 迭代训练
        if epoch < 80:
            learning_rate = learning_rate
        elif epoch < 120:
            learning_rate = learning_rate * 0.1
        else:
            learning_rate = learning_rate * 0.01
        for param_group in optimizer.param_groups:
            param_group['learning_rate'] = learning_rate

        train(epoch)
        loss = test()

    now = time.gmtime(time.time() - start_time)
    # 打印训练时间
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))
    torch.save(model.state_dict(), '../pth/model.pth')

def main():
    model = Vgg() # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth', map_location='cpu'))
    return model

#start()