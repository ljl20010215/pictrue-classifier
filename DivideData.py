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