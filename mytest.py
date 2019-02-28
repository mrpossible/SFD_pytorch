from __future__ import print_function

import torch
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from numpy import array
from numpy import argmax
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
import torchvision
import torchvision.transforms as transforms

import os,sys,cv2,random,datetime,time,math
import pandas as pd
import argparse
import numpy as np
from net_s3fd import *
from bbox import *
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from PIL import Image
test_data = "/home/user/datasets/img_align_celeba/10_class/final_10_class_test.csv"
test_path = "/home/user/datasets/img_align_celeba/10_class/test/"
img_ext = ".jpg"


def transform(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)

    # img = img.reshape((1,)+img.shape)
    img = torch.from_numpy(img).float()
    return img

# transform = transforms.Compose([
#     # you can add other transformations in this list
#     transforms.CenterCrop((128, 128)),
#     transforms.transpose(2, 0, 1),
#     transforms.ToTensor(),
#
# ])

testset = torchvision.datasets.ImageFolder(test_path, transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

def revert_transform(img):
    img = img.to
def transform(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (128, 128))
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)

    # img = img.reshape((1,)+img.shape)
    img = torch.from_numpy(img).float()

    img = img.view((1,) + img.shape[1:])
    return img


def imshow1(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (0, 1, 2))
    npimg = npimg + np.array([104, 117, 123])  # unnormalize
    # npimg = npimg + np.array([104, 117, 123])
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if os.path.isfile('faceRecog.saved.model'):
    num_classes = 10
    myModel = s3fd(num_classes)
    for param in myModel.parameters():
        param.requires_grad = False
    myModel.fc_1 = nn.Linear(1024, num_classes)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, myModel.parameters()), lr=0.0001, momentum=0.9)

    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9)
    print("=> loading checkpoint")
    checkpoint = torch.load('faceRecog.saved.model')
    myModel.load_state_dict(checkpoint['model_state_dict'])

    #
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    myModel.eval()


    print("=> loaded checkpoint finish")

    myModel = myModel.cuda()

    ###########
    # Test the model
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(4)))

    images = images.cuda()
    # let see what the NN thinks these examples above are:
    output = myModel(images)
    print("test Image1 - ", output)

    _, predicted = torch.max(output, 1)

    print('Predicted: ', ' '.join('%5s' % predicted[j]
                                  for j in range(4)))


else:
    print("=> no checkpoint found")

# testImage1 = transform('data/Test/TestCeleb_4/25-FaceId-0.jpg')
# testImage2 = transform('data/Test/TestCeleb_4/26-FaceId-0.jpg')
# testImage3 = transform('data/Test/TestCeleb_4/27-FaceId-0.jpg')
# testImage4 = transform('data/Test/TestCeleb_10/25-FaceId-0.jpg')
# testImage5 = transform('data/Test/TestCeleb_10/26-FaceId-0.jpg')
# testImage6 = transform('data/Test/TestCeleb_10/24-FaceId-0.jpg')
#
# output1 = myModel(testImage1)
# output2 = myModel(testImage2)
# output3 = myModel(testImage2)
# output4 = myModel(testImage4)
# output5 = myModel(testImage5)
# output6 = myModel(testImage6)
# print("testImage1 - ",output1)
# print("testImage2 - ",output2)
# print("testImage3 - ",output3)
# print("testImage1 - ",output4)
# print("testImage2 - ",output5)
# print("testImage3 - ",output6)
