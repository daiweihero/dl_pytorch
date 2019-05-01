# ---------------------------------------------------------------------------- #
# 作者：代蔚
# 创建时间：2019年5月1日
# 第二章 神经网络的数学基础
# ---------------------------------------------------------------------------- #

import torch
from torch.utils.data import DataLoader
from torchvision import datasets  # 包含MNIST、CIFAR-10等常用数据集
from torchvision import transforms
import torch.nn as nn

# ch2.1 代码清单 2-1 ---------------------------------------------------------- #
data_path = '../../data'
train = datasets.MNIST(  # 返回MNIST对象
    root=data_path,  # 本地数据集存放路径
    train=True,  # True加载训练集，False加载测试集
    transform=transforms.ToTensor(),  # 是否对数据预处理
    download=True  # 如果本地数据集不存在，则从网络下载
)
test = datasets.MNIST(
    root=data_path, train=False, transform=transforms.ToTensor())

# 查看数据
train.data.shape  # [60000, 28, 28]
len(train.data)  # 60000
train.targets.shape  # [60000]

# 小批量加载数据
train_loader = DataLoader(dataset=train, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=128, shuffle=False)


# ch2.1 代码清单 2-2 网络架构 ------------------------------------------------- #
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


model = Net(28 * 28, 500, 10)

# ch2.1 代码清单 2-3 编译步骤 ------------------------------------------------- #
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# ch2.1 代码清单 2-4 准备图像数据 --------------------------------------------- #
# 只是等价于keras的写法，实际并不运行以下四行代码
# train_imagess = train.data.reshape((60000, 28 * 28))
# train_imagess = train_imagess.float() / 255
# test_imagess = test.data.reshape((10000, 28 * 28))
# test_imagess = test_imagess.float() / 255

# ch2.1 代码清单 2-5 准备标签 ------------------------------------------------- #
# CrossEntropyLoss()会自动转化one-hot编码
# 训练网络
for epoch in range(5):  # epochs=5
    for images, labels in train_loader:
        # 准备图像数据放在这一行
        images = images.reshape((-1, 28*28)).float() / 255
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}] loss: {}'.format(epoch + 1, 5, loss.item()))

# 检查测试集上的性能
with torch.no_grad():  # 不需要反向传播，禁用自动梯度功能
    correct = 0  # 预测正确的数量
    total = len(test.targets)
    for images, labels in test_loader:
        images = images.reshape((-1, 28*28)).float() / 255
        outputs = model.forward(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
    print('test accuracy: {}%'.format(correct / total * 100))

# 总结 ------------------------------------------------------------------------ #
# pytorch的代码比keras多了很多
# tytorch的模型在我的电脑上运行多次，每次的精确率大约在92.72%左右，比起书中97.85%差了很多