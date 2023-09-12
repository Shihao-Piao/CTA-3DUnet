import numpy
import nibabel as nib
import matplotlib.pyplot  as plt
import torch
import torch.nn as nn
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils import data

from net_model import UNet3D, EncoderUNet3D, DecoderUNet3D
from utils import NewDiceLoss, cal_dice_loss, MyDataset_UNet, check_and_create_path, Logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os
# 只使用一个线程
os.environ["OMP_NUM_THREADS"] = "1"

data_path = '../heart_data_nii/label/11.nii'
pred_path = '2-prediction/0.nii'

epi_img = nib.load(data_path)
epi_img = nib.load(pred_path)
print(epi_img.shape)
epi_img_data = epi_img.get_data()
print(epi_img_data.shape)


def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice, cmap="gray", origin="lower")


def print_data(data,start,end,layer):
    tmp_data = data[:,:,layer]
    tmp_loop = end - start
    for i in range(tmp_loop):
        for j in range(tmp_loop):
            print(tmp_data[start+i][start+j],end=' ')
        print('\n')


print_data(epi_img_data,150,220,100)
slice_0 = epi_img_data[52, :, :]
slice_1 = epi_img_data[:, 13, :]
slice_2 = epi_img_data[:, :, 100]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
plt.show()

data = [
    [[1,2],[5,6]],
    [[3,4],[7,8]]
]
input = autograd.Variable(torch.randn(1, 2, 2, 2, 2))
print(input)

conv =nn.ConvTranspose3d(2, 1, kernel_size=2, stride=2)

out = conv(input)
print(out)


import matplotlib.pyplot as plt
import numpy as np

x = []

for i in range(300):
    x.append(i+1)


y = [
0.9973,0.9967,0.9945,0.9882,0.9536,0.9202,0.7843,0.7270,0.7722,0.7023,
0.6646,0.6912,0.6676,0.6279,0.6064,0.6286,0.5852,0.5340,0.5454,0.5409,
0.5500,0.5421,0.5732,0.5437,0.4933,0.5087,0.4679,0.4886,0.4189,0.4945,
0.4896,0.4195,0.4369,0.4730,0.4208,0.3805,0.3936,0.4323,0.4544,0.3898,
0.3932,0.4542,0.4692,0.3858,0.3744,0.4107,0.3851,0.3510,0.3798,0.3515,
0.3863,0.3284,0.4246,0.3464,0.4219,0.3066,0.3671,0.3106,0.3750,0.3767,
0.3041,0.3094,0.3102,0.2896,0.3213,0.2987,0.3885,0.3864,0.3963,0.3415,
0.3478,0.3355,0.3239,0.3488,0.3058,0.3345,0.2948,0.3005,0.3256,0.3234,
0.3790,0.3208,0.2705,0.3626,0.2908,0.3045,0.3037,0.3294,0.2938,0.3114,
0.2549,0.3652,0.3056,0.3494,0.2826,0.2957,0.3202,0.3085,0.3041,0.2702,
0.3039,0.2577,0.2721,0.2603,0.2742,0.2833,0.2886,0.2900,0.2734,0.2608,
0.2589,0.2704,0.2471,0.3095,0.2995,0.2446,0.2685,0.3059,0.3070,0.2724,
0.2412,0.2350,0.2230,0.2746,0.3130,0.3126,0.3173,0.2557,0.2443,0.3082,
0.2733,0.2733,0.2763,0.2905,0.2858,0.2648,0.2593,0.2623,0.2559,0.2515,
0.2664,0.3046,0.2577,0.2770,0.2584,0.2816,0.2803,0.2602,0.2395,0.2581,
0.2473,0.2382,0.2739,0.2413,0.2414,0.2580,0.2610,0.2662,0.2358,0.2242,
0.2533,0.2506,0.2320,0.2474,0.2446,0.2480,0.2332,0.2189,0.2320,0.2591,
0.2339,0.2317,0.2466,0.2751,0.2135,0.2431,0.2291,0.2178,0.2340,0.2128,
0.2103,0.2288,0.2553,0.2478,0.2280,0.1988,0.2144,0.2165,0.2123,0.2105,
0.1974,0.2151,0.2379,0.2158,0.2066,0.2215,0.2255,0.2217,0.2405,0.2414,
0.2038,0.1957,0.2013,0.1990,0.2040,0.2111,0.2097,0.2078,0.1907,0.2267,
0.2132,0.1858,0.1896,0.1960,0.2189,0.1865,0.2013,0.1786,0.2050,0.2116,
0.1740,0.1904,0.1704,0.1912,0.1789,0.1796,0.1787,0.1693,0.2039,0.1979,
0.1943,0.2053,0.1758,0.1780,0.2014,0.1955,0.2091,0.1898,0.1908,0.1787,
0.1671,0.1872,0.1590,0.1851,0.2025,0.1601,0.1847,0.1797,0.1745,0.1667,
0.1643,0.1787,0.1678,0.1619,0.1650,0.2040,0.1964,0.1671,0.1658,0.1855,
0.1597,0.1592,0.1593,0.1576,0.1652,0.1550,0.1553,0.1678,0.1635,0.1582,
0.1615,0.1709,0.1732,0.1597,0.1769,0.1574,0.1624,0.1578,0.1565,0.1663,
0.1617,0.1681,0.1513,0.1555,0.1635,0.1602,0.1549,0.1390,0.1637,0.1506,
0.1718,0.1571,0.1495,0.1589,0.1450,0.1663,0.1636,0.1505,0.1505,0.1505
]


# 请求一个新的figure对象
plt.figure()
# 第一个是横坐标的值，第二个是纵坐标的值
plt.plot(x, y)

# 设置轴线的lable（标签）
plt.xlabel("Epoch")
plt.ylabel("DiceLoss")

plt.show()



'''
############################################################################
data_dir = '../heart_data_nii/train/'
label_dir = '../heart_data_nii/label/'

from sklearn.model_selection import train_test_split
patient_list = os.listdir(data_dir)
train_list_initail, test_list = train_test_split(patient_list[:], test_size=0.9, random_state=12)
train_list, val_list = train_test_split(train_list_initail, test_size=0.5, random_state=12)
print(len(train_list), len(val_list), len(test_list))

train_set = MyDataset_UNet(train_list, data_dir, label_dir, valid_flag=False)
train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
valid_set = MyDataset_UNet(val_list, data_dir, label_dir, valid_flag=True)
valid_loader = data.DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1)



for index, (img, label) in enumerate(train_loader):

    print(index,":",img.shape)
    #print(img)

    # img = img.cuda()
    img = img.to(device)
    print(index, ":", img.shape)

    # label = label.cuda()
    label = label.to(device)
    print(index, ":", label.shape)


for index, (img, label) in enumerate(valid_loader):

    print(index,":",img.shape)
    #print(img)

    # img = img.cuda()
    img = img.to(device)
    print(index, ":", img.shape)

    # label = label.cuda()
    label = label.to(device)
    print(index,":",label.shape)


'''