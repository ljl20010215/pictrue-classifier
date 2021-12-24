from icecream import ic
from torchvision import datasets
from tqdm import tqdm
import os

train_data = datasets.MNIST(root="./data/", train=True, download=True)
test_data = datasets.MNIST(root="./data/", train=False, download=True)
saveDirTrain = 'D:\\mnist_images\\Test'
saveDirTest = 'D:\\mnist_images\\Train'

if not os.path.exists(saveDirTrain):
    os.mkdir(saveDirTrain)
if not os.path.exists(saveDirTest):
    os.mkdir(saveDirTest)

ic(len(train_data), len(test_data))
ic(train_data[0])
ic(train_data[0][0])


def save_img(data, save_path):
    for i in tqdm(range(len(data))):
        img, label = data[i]
        img.save(os.path.join(save_path, str(i) + '-label-' + str(label) + '.png'))


save_img(train_data, saveDirTrain)
save_img(test_data, saveDirTest)
