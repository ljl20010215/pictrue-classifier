# pictrue-classifier
It's a good tool to help you do a job that you need classify many pictrues into different classes.It's completed by  Convolutional Neural Network.Maybe it's accuracy isn't high enough.But I believe you need have a try!!!
## 程序功能

实现一个图片集当中的图片分类。

## 窗口界面及功能介绍

<img src="https://pic.imgdb.cn/item/61c332f32ab3f51d9134e163.png" alt="程序窗口" style="zoom:67%;" />

窗口主要由Treeview、Label、Button、Entry、Combobox等组件组成。

**Treeview**主要实现文件夹以及文件的展示，可拖拽文件进入Treeview。

**第一个Entry**可输入文件（夹）路径，或者通过旁边的按钮打开文件管理，选中文件后点击确定，也能实现获取该文件路径并显示到该Entry。

**第一个Combobox**可实现选择要分类的数目，所选的数目要满足pkl文件一定的格式，后面会详细说明。

**第二个Combobox**可实现选择训练次数，训练次数越多，精度可能就会有所提高。

**第二个Entry**可输入划分训练集时的比率。

**三个按钮**分别实现图片分类、训练图片集、划分图片集这三个功能，点击任意一个按钮都会生成一个新的线程来执行功能，在执行该功能时所有按钮都是禁用状态。

**右下角的Label**用来显示执行每个任务时的进程。

## 项目结构与代码注解

<img src="https://pic.imgdb.cn/item/61c341c62ab3f51d91392255.png" alt="项目结构" style="zoom:50%;" />

### 划分数据集

**需要参数：**将目标数据集（包含多个文件夹，文件夹内包含图片）拖入Treeview并选中，选择要将划分好的数据保存到哪（即第一个Entry中的路径）

___

在点击Divide Data按钮后，右下角会显示划分进度，划分完成后会在目标路径生成一个**Data**文件夹，里面包含了**Lib，Train，Test**三个文件夹，Train和Test文件夹内包含图片，Lib文件夹内会保存训练后的pkl文件和一张训练过程中准确率和损失变化的曲线图。划分结束后，Treeview中也会出现Data文件夹。

```python
# 2021/12/9 14:44

from Global import *
import os
import random
from shutil import copy

# data_path=sys.argv[1]           # ImageNet 路径

def create_data_file(goal_path):
    if not os.path.exists(os.path.join(goal_path,'Data')):
        os.mkdir(os.path.join(goal_path,'Data'))
    if not os.path.exists(os.path.join(goal_path,'Data','Train')):
        os.mkdir(os.path.join(goal_path,'Data','Train'))
    if not os.path.exists(os.path.join(goal_path,'Data','Test')):
        os.mkdir(os.path.join(goal_path,'Data','Test'))
    if not os.path.exists(os.path.join(goal_path,'Data','Lib')):
        os.mkdir(os.path.join(goal_path,'Data','Lib'))

# 按照预设的比例进行分割数据集
def GetTrainAndTest(data_path,train_rate,goal_path):
    create_data_file(goal_path)
    DataSet_Names=[name for name in os.listdir(data_path) if not name.endswith('.tar')]
    for x,DataSet_name in enumerate(DataSet_Names):
        g_values.progress_rate2=(x,len(DataSet_Names))              # 设置进度
        DataSet_path=os.path.join(data_path,DataSet_name)
        image_names=[name for name in os.listdir(DataSet_path)]              # 图片名称
        train_images=random.sample(image_names,int(train_rate * len(image_names)))         # 随机抽取图片
        print(f'Getting Train DataSet {DataSet_name}....')
        tnp=os.path.join(goal_path,'Data','Train',DataSet_name)        # 训练集中的dog类路径
        if not os.path.exists(tnp):
            os.mkdir(tnp)
        for index,image_name in enumerate(train_images):
            g_values.progress_rate1=(index,len(train_images),'TRAIN')           # 设置进度
            copy(os.path.join(DataSet_path,image_name),os.path.join(goal_path,'Data','Train',DataSet_name,image_name))
        test_images=[name for name in os.listdir(DataSet_path) if name not in(train_images)]        # 获得测试集
        print(f'Getting Test DataSet {DataSet_name}....')
        ttp=os.path.join(goal_path,'Data','Test',DataSet_name)
        if not os.path.exists(ttp):
            os.mkdir(ttp)
        for index,image_name in enumerate(test_images):
            g_values.progress_rate1=(index,len(test_images),'TEST')            # 设置进度
            copy(os.path.join(DataSet_path,image_name),os.path.join(goal_path,'Data','Test',DataSet_name,image_name))
    g_values.flag=1

# def getDisplay(data_path,number,goal_path):     # number 为每一个图片集抽取图片的数量
#     DataSet_Names = [name for name in os.listdir(data_path) if not name.endswith('.tar')]
#     for DataSet_name in DataSet_Names:
#         DataSet_path=os.path.join(data_path,DataSet_name)
#         image_names = [name for name in os.listdir(DataSet_path)]
#         samples_name= [temp for temp in random.sample(image_names,number)]
#         for index,name in enumerate(samples_name):
#             copy(os.path.join(DataSet_path,name),os.path.join(goal_path,'Data','display',name))

# getDisplay(100)
```

### 训练数据集

**需要参数：**选中Data（如果是在启动程序后立即训练，则将划分后得到的文件夹Data拖入Treeview），选择训练次数

___

在点击Start to train按钮后，右下角会显示训练进度，**进度组成为：该次训练进度，训练次数进度，训练至此的最大准确率**，训练过程中，Lib文件夹下会生成pkl文件，该文件名格式如下：**best_model\_训练次数\_该文件准确率\_分类数.pkl**，该文件必须以此格式命名，"best_model"为可更改部分。训练完成后，Lib文件夹下保存的pkl文件为本次训练的最佳准确率文件，**plot.png**为准确率以及loss在训练过程中的变化过程。

```python
# 2021/12/11 8:20
import torch
import time
from torch import nn
from Global import g_values
from CNN import MyAlexNet
from torch.optim import lr_scheduler
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 训练数据
class Trainer:
    def __init__(self,ROOT_TRAIN,ROOT_TEST,GOAL_PATH):
        self.ROOT_TRAIN=ROOT_TRAIN              # 训练集目录，Train
        self.GOAL_PATH = GOAL_PATH              # 分类目标目录
        self.ROOT_TEST=ROOT_TEST                # 测试集目录，Test
        self.pic_class=[cls for cls in os.listdir(ROOT_TRAIN)]          # 将训练集的目录名记录下来以便在分类时使用

        # 将图像的像素值归一化到【-1， 1】之间
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图片大小
            transforms.RandomVerticalFlip(),  # 随机变换
            transforms.ToTensor(),  # 转换为张量
            self.normalize])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize])

        self.train_dataset = ImageFolder(self.ROOT_TRAIN, transform=self.train_transform)  # 获取到已经分类完成的数据集
        self.test_dataset = ImageFolder(self.ROOT_TEST, transform=self.test_transform)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=48, shuffle=True)  # 将数据每32个一批进行训练，且每次训练时将数据打乱
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=48, shuffle=True)


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'            # 判断是否可以使用GPU进行训练

        self.model = MyAlexNet(num=len(os.listdir(ROOT_TRAIN))).to(self.device)     # 建立模型，分类数由训练集目录数目决定

        # 定义一个损失函数
        self.loss_fn = nn.CrossEntropyLoss()

        # 定义一个优化器
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)      # 随机梯度下降的优化器模型，学习率为0.01,momentum为冲量，可以对学习率进行矫正,起到加速、减速的目的

        # 学习率每隔10轮变为原来的0.5,使梯度下降过程越来越精细
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        # 开始训练
        self.loss_train = []
        self.acc_train = []
        self.loss_test = []
        self.acc_test = []


    # 定义训练函数
    def train(self):
        loss, current, n = 0.0, 0.0, 0
        for batch, (x, y) in enumerate(self.train_dataloader):
            g_values.progress_rate1=(batch,len(self.train_dataloader),'Train')          # 记录进度
            image, y = x.to(self.device), y.to(self.device)
            output = self.model(image)
            cur_loss = self.loss_fn(output, y)              # 计算损失
            _, pred = torch.max(output, axis=1)                     # 返回每行最大值的索引
            cur_acc = torch.sum(y == pred) / output.shape[0]          # 计算准确率

            # 反向传播
            self.optimizer.zero_grad()               # 梯度归0
            cur_loss.backward()             # 计算每个参数的梯度值
            self.optimizer.step()                # 更新每个参数
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n+1

        train_loss = loss / n               # 计算平均损失
        train_acc = current / n             # 计算平均准确度
        print('train_loss ' + str(train_loss))
        print('train_acc ' + str(train_acc))
        return train_loss, train_acc

    # 定义一个验证函数
    def test(self):
        # 将模型转化为验证模型
        self.model.eval()
        loss, current, n = 0.0, 0.0, 0
        with torch.no_grad():               # 不用进行梯度下降
            for batch, (x, y) in enumerate(self.test_dataloader):
                g_values.progress_rate1=(batch,len(self.test_dataloader),'Test')
                image, y = x.to(self.device), y.to(self.device)
                output = self.model(image)              # 进行预测
                cur_loss = self.loss_fn(output, y)
                _, pred = torch.max(output, axis=1)
                cur_acc = torch.sum(y == pred) / output.shape[0]
                loss += cur_loss.item()
                current += cur_acc.item()
                n = n + 1

        test_loss = loss / n
        test_acc = current / n
        print('test_loss ' + str(test_loss))
        print('test_acc ' + str(test_acc))
        return test_loss, test_acc

    # 定义画图函数
    def matplot_loss(self,train_loss, val_loss):
        plt.plot(train_loss, label='train_loss')
        plt.plot(val_loss, label='test_loss')
        plt.legend(loc='best')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title("训练集和验证集loss值对比图")
        # plt.savefig(os.path.join(self.GOAL_PATH,'loss_plot.png'))

    def matplot_acc(self,train_acc, val_acc):
        plt.plot(train_acc, label='train_acc')
        plt.plot(val_acc, label='test_acc')
        plt.legend(loc='best')
        plt.ylabel('acc/loss')
        plt.xlabel('epoch')
        plt.title("训练集和验证集acc值对比图")
        plt.savefig(os.path.join(self.GOAL_PATH,'plot.png'))


    def main_train(self,epoch):
        min_acc = 0
        old_acc=0
        for t in range(epoch):
            g_values.progress_rate2=(t,epoch)
            self.lr_scheduler.step()
            print(f"epoch{t+1}\n-----------")
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()

            self.loss_train.append(train_loss)
            self.acc_train.append(train_acc)
            self.loss_test.append(test_loss)
            self.acc_test.append(test_acc)

            # 保存最好的模型权重
            if test_acc >min_acc:
                min_acc = test_acc
                g_values.acc=test_acc
                print(f"save best model, 第{t+1}轮")
                # 把旧的文件删除
                if os.path.exists(os.path.join(self.GOAL_PATH,f'best_model_{epoch}_{format(old_acc,"0.3f")}_{len(self.pic_class)}.pkl')):
                    os.remove(os.path.join(self.GOAL_PATH,f'best_model_{epoch}_{format(old_acc,"0.3f")}_{len(self.pic_class)}.pkl'))
                old_acc = test_acc
                torch.save(self.model.state_dict(), os.path.join(self.GOAL_PATH,f'best_model_{epoch}_{format(test_acc,"0.3f")}_{len(self.pic_class)}.pkl'))
                g_values.pkl_path=os.path.join(self.GOAL_PATH,f'best_model_{epoch}_{format(test_acc,"0.3f")}_{len(self.pic_class)}.pkl')
                g_values.pkl_name=f'best_model_{epoch}_{format(test_acc,"0.3f")}_{len(self.pic_class)}.pkl'

        g_values.flag=1
        self.matplot_loss(self.loss_train, self.loss_test)
        self.matplot_acc(self.acc_train, self.acc_test)
        plt.clf()
        print('Done!')
        return self

    # 训练、测试完进行reset
    def reset(self):
        self.loss_train.clear()
        self.loss_test.clear()
        self.acc_train.clear()
        self.acc_test.clear()
        return self.pic_class
```

### 对数据集分类

**需要参数：**待分类目标文件夹（该文件夹下保存了待分类图片），选择的分类数目，Treeview中选中的pkl文件（如果Treeview中没有pkl文件则需要自行拖入）

___

分类的过程中，右下角显示进度，**如果在分类前执行了以上两步**，则在目标文件夹下会生成与训练集当中数据类名称一样的几个文件夹，**否则**会生成cls\_0，cls\_1等文件夹并以该名称代表数据类名称进行分类。分类完成后，原目标文件夹下的图片会被分类到生成的类文件夹下，完成分类。

```python
# 2021/12/11 21:41
import os

import torch
import numpy
import shutil
from Global import g_values
from PIL import Image
from CNN import MyAlexNet
from torchvision import transforms
from torch.utils.data import Dataset



class Mydataset(Dataset):
    def __init__(self,ROOT_PATH):
        self.ROOT_PATH=ROOT_PATH
        self.images=[temp for temp in os.listdir(self.ROOT_PATH) if temp.endswith('JPEG')]      # 图片名称
    def __getitem__(self, item):
        img=Image.open(os.path.join(self.ROOT_PATH,self.images[item]))
        resize=transforms.Resize([224,224])
        to_tensor=transforms.ToTensor()
        re_img=resize(img)
        tensor_img=to_tensor(re_img)
        return os.path.join(self.ROOT_PATH,self.images[item]),tensor_img
    def __len__(self):
        return len(self.images)

class classifier:
    def __init__(self,GOAL_PATH,MODEL_PATH,cls_name,class_num):
        self.GOAL_PATH=GOAL_PATH            # 待分类图片的路径
        self.MODEL_PATH=MODEL_PATH          # best模型路径
        self.CLS=cls_name                   # 分类名
        self.dataset=Mydataset(self.GOAL_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MyAlexNet(num=class_num).to(self.device)
        self.model.load_state_dict(torch.load(self.MODEL_PATH))


    def classify(self):
        # 创建分类的目录
        for cls in self.CLS:
            path = os.path.join(self.GOAL_PATH, cls)
            if not os.path.exists(path):
                os.mkdir(path)
        # 进入到验证阶段
        self.model.eval()
        for i in range(len(self.dataset)):
            g_values.progress_rate1=(i,len(self.dataset),'classifying')
            image_path,img=self.dataset[i]
            if img.size()[0] != 3:                        # 如果一张图片不是三通道，则将它转换为三个通道
                img=torch.tensor(numpy.array([img.numpy()[0],img.numpy()[0],img.numpy()[0]]))
            img = torch.tensor(numpy.array([img.numpy()])).to(self.device)
            with torch.no_grad():
                pred = self.model(img)
                predicted= self.CLS[torch.argmax(pred[0])]
                shutil.move(image_path,os.path.join(self.dataset.ROOT_PATH,predicted))
        g_values.flag=1

```

## 总结

该程序主要通过卷积神经网络识别图片并对图片进行分类，这是一种**监督学习**，也就是说，我们一开始就要知道图片集当中有哪些类（标签），然后再进行分类。该程序的**优越性**在于只需要pkl文件就可以对一个图片集进行分类，如果可以建立一个pkl文件库，就可以很方便得实现各种图片的分类。但实际上，一个图片集当中所包含的类型大概率是未知的，并且要pkl文件库当中包含所有物品类别全排列的类型也是不太现实的，也就是说，该程序的实际应用具有非常大的**局限性**。

