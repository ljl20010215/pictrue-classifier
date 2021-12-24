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
