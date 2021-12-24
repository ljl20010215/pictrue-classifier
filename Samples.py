# 2021/12/15 17:03
import os
import random
from tqdm import tqdm
from shutil import copy

# 从图片集中随机抽取图片形成一个测试集
def sample(path,goal_path,rate,rname):
    data_name=[n for n in os.listdir(path) if not n.endswith('.tar')]
    data_path=[os.path.join(path,n) for n in os.listdir(path) if not n.endswith('.tar')]
    for index, d_p in enumerate(data_path):
        l=len(os.listdir(d_p))
        samples=random.sample([os.path.join(d_p,x) for x in os.listdir(d_p)],int(rate*l))
        for x,sm in tqdm(enumerate(samples)):
            if not rname:
                copy(sm,os.path.join(goal_path,data_name[index]+f'_{x}.JPEG'))
            else:
                copy(sm,os.path.join(goal_path,str(random.randint(0,10000))+f'_{x}.JPEG'))

if __name__=='__main__':
    # sample('D:\\mnist_images\\Train','D:\\mydataset\\num_class',0.8,False)
    # sample('D:\\mydataset\data_cat_dog\\Train', 'D:\\mydataset\\test_2_dog_cat', 0.75, False)
    sample('D:\\BaiduNetdiskDownload\\Data\\Train','D:\\mydataset\\plant_class',0.7,False)