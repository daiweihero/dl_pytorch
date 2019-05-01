import torch
# import torchvision
from torch import nn
import numpy as np

# ch3.4 代码清单 3-1 加载IMDB数据集 ------------------------------------------- #
path = '../.keras/datasets/imdb.npz'  # 自行下载imdb.npz文件
imdb_np = np.load(path)


def max_words(lst, num_words=10000):
    # 删除imdb.npz文件中数字大于10000的值
    lstlst = []
    for i in lst:
        i = [j for j in i if j < num_words]
        lstlst.append(i)
    return lstlst


def load_imdb(data):
    test_data = max_words(data['x_test'])
    test_labels = data['y_test']

    train_data = max_words(data['x_train'])
    train_labels = data['y_train']

    return train_data, train_labels, test_data, test_labels


train_data, train_labels, test_data, test_labels = load_imdb(imdb_np)


# ch3.4 代码清单 3-2 one-hot -------------------------------------------------- #
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# ch3.4 代码清单 3-3 模型定义 ------------------------------------------------- #
class Net(torch.nn.Module):
    def __init__(self, i_size, h1_size, h2_size, o_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(i_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, o_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


model = Net(10000, 16, 16, 1)

# ch3.4 代码清单 3-4 编译模型 3-5, 3-6 ---------------------------------------- #
criterion = nn.BCELoss()  # 二进制交叉熵
optimizer = torch.optim.RMSprop(model.parameters())

# ch3.4 代码清单 3-7 留出验证集 ----------------------------------------------- #
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# ch3.4 代码清单 3-8 训练模型 ------------------------------------------------- #
batch_size = 512
partial_x_train = torch.from_numpy(partial_x_train).float()
partial_y_train = torch.from_numpy(partial_y_train)
m = len(partial_y_train)

for epoch in range(4):
    for i in range(m // batch_size):
        outputs = model.forward(
            partial_x_train[i * batch_size:(i + 1) * batch_size])
        loss = criterion(outputs,
                         partial_y_train[i * batch_size:(i + 1) * batch_size])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/20] loss: {}'.format(epoch + 1, loss.item()))
