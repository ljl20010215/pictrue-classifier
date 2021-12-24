# 2021/12/9 14:46
# 从data.txt中获取到的类型对应的数据集

import sys
import os
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)

from Exception.MyException import MoreParameter,LessParameter

if len(sys.argv) < 2 :
    raise MoreParameter()
if len(sys.argv) > 2 :
    raise LessParameter()
#输入你要查找的字符串
findSomething=sys.argv[1]

data_class_dict={}
with open('./data.txt') as file:
    str=file.read()
str=str[1:-1]
list=str.split(',')
for index in range(len(list)-1):
    if index & 1 :
        pass
    else:
        dataset_name=list[index][list[index].find('[')+2:-1]
        class_name=list[index+1][2:-2]
        data_class_dict[class_name]=dataset_name
# print(data_class_dict)

# ==========================================================

SomethingLine=0;
#输入txt文件
f = open('./imagenet-classes.txt', 'r', encoding='utf-8')
for lines in f.readlines():
    SomethingLine=SomethingLine+1;
    if lines.find(findSomething)!=-1:
        lines=lines.strip()
        lines=[t.strip() for t in lines.split(',')]
        lines=[word.replace(' ','_') for word in lines]
        lines=[x for x in lines if x.find(findSomething)!=-1]
        for item in lines:
            if item in data_class_dict:
                print(item,'-->',data_class_dict[item],sep=' ')
f.close()
