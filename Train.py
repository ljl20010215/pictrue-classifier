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

    def matplot_acc(self,train_acc, val_acc,max_acc,epoch):
        plt.plot(train_acc, label='train_acc')
        plt.plot(val_acc, label='test_acc')
        plt.legend(loc='best')
        plt.ylabel('acc/loss')
        plt.xlabel('epoch')
        plt.title("训练集和验证集acc值对比图")
        plt.savefig(os.path.join(self.GOAL_PATH,f'plot_{max_acc}_{epoch}.png'))


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
        self.matplot_acc(self.acc_train, self.acc_test,max_acc=min_acc,epoch=epoch)
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