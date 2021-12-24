# 2021/12/9 15:02

import sys
import os

ImageNet_path=sys.argv[1]
DataName_path=[name for name in os.listdir(ImageNet_path) if not name.endswith('.tar')]
for DataSet_Name in DataName_path:
    path=os.path.join(ImageNet_path,DataSet_Name)
    images=[os.path.join(ImageNet_path,DataSet_Name,image_path) for image_path in os.listdir(path)]
    for index,image in enumerate(images):
        New_Name=DataSet_Name+'_'+str(index)+'.JPEG'
        os.renames(image,os.path.join(ImageNet_path,DataSet_Name,New_Name))
