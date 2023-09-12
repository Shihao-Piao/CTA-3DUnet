import numpy as np
import scipy.io as scio
import nibabel as nib
import matplotlib.pyplot  as plt
import numpy as np
import math
import SimpleITK as sitk
import torch
import os


data_path = '../heart_data_nii/label/11.nii'
pred_path = '2-prediction/0.nii'

epi_img = nib.load(data_path)
epi_img = nib.load(pred_path)
print(epi_img.shape)
epi_img_data = epi_img.get_data()
shape = epi_img_data.shape
print(epi_img_data.shape)
#print(epi_img_data[300,300,100])


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


def get_direction(x,y,z):
    direction = []
    select = [-1,0,1]
    for i in select:
        for j in select:
            for k in select:
                if (x*i + y*j +z*k == 0) and (i**2 + j**2 + k**2 != 0):
                    direction.append([i,j,k])
    return direction

def judge_edge(x,y,z,s = shape):
    if x>=0 and y>=0 and z>=0 and x<=s[0] and y<=s[1] and z<=s[2]:
        return 1
    else:
        return 0



#print_data(epi_img_data,150,220,100)
#slice_0 = epi_img_data[52, :, :]
#slice_1 = epi_img_data[:, 13, :]
#slice_2 = epi_img_data[:, :, 100]
#show_slices([slice_0, slice_1, slice_2])
#plt.suptitle("Center slices for EPI image")
#plt.show()

mat_path = '0.mat'
data = scio.loadmat(mat_path)
data = data['Line']

###################################
#for i in range(len(data)):
#    print(len(data[i][0]))
#print(data[-1][0][0])
print(len(data))
print(data[0])

radius_list = []

def pred(image,point):
    for i in range(point[2]-4,point[2]+5):
        for j in range(point[1] - 2, point[1] + 3):
            for k in range(point[0] - 2, point[0] + 3):
                image[i,j,k] = 1



for i in range(len(data)):
    line = np.array(data[i][0],dtype='int32')
    tmp_radius = []
    for j in range(1,len(line)):
        x = line[j][0] - line[j-1][0]
        y = line[j][1] - line[j-1][1]
        z = line[j][2] - line[j-1][2]
        direction = get_direction(x,y,z)
        radius = 0
        for k in range(len(direction)):
            direct = direction[k]
            dx = direct[0]
            dy = direct[1]
            dz = direct[2]
            rx = line[j-1][0]
            ry = line[j - 1][1]
            rz = line[j - 1][2]
            xx = rx + dx
            yy = ry + dy
            zz = rz + dz
            while(judge_edge(xx,yy,zz) and epi_img_data[xx,yy,zz] == 1):
                radius += math.sqrt(dx**2+dy**2+dz**2)
                xx += dx
                yy += dy
                zz += dz
        tmp_radius.append(round(radius,3))
    radius_list.append(tmp_radius)


print("一条线段上的半径计算结果：",radius_list[0],'\n')
#print(radius_list[8],'\n')
#print(radius_list[9],'\n')
#print(radius_list[-4],'\n')
#print(len(radius_list[0]))

warn_list_idx = []
warn_list_radius = []
para = 0.75
for i in range(len(radius_list)):
    warn = []
    warn_radius =[]
    if len(radius_list[i]) >= 5:
        tmp = []
        for j in range(len(radius_list[i])-4):
            tmp.append(radius_list[i][j])
            tmp.append(radius_list[i][j+1])
            tmp.append(radius_list[i][j+2])
            tmp.append(radius_list[i][j+3])
            tmp.append(radius_list[i][j+4])
            mean = np.mean(tmp)
            for k in range(5):
                if mean <= 0.01:
                    pass
                    '''
                    idx = j+k
                    if idx not in warn:
                        warn.append(idx)
                        warn_radius.append(0)
                    '''
                else:
                    if (mean - tmp[k])/mean >= para:
                        #print(tmp[k])
                        idx = j+k
                        if idx not in warn:
                            warn.append(idx)
                            warn_radius.append(radius_list[i][idx])
            tmp =[]

    else:
        mean = np.mean(radius_list[i])
        for m in range(len(radius_list[i])):
            if mean <= 0.01:
                pass
                '''
                if m not in warn:
                    warn.append(m)
                    warn_radius.append(0)
                    '''
            else:
                if (mean - radius_list[i][m])/mean >= para:
                    if m not in warn:
                        warn.append(m)
                        warn_radius.append(radius_list[i][m])

    warn_list_idx.append(warn)
    warn_list_radius.append(warn_radius)

print(warn_list_radius)
print(warn_list_idx)


warn_point = []
for i in range(len(radius_list)):
    point = []
    line = np.array(data[i][0],dtype='int32')
    idxs = warn_list_idx[i]
    if len(idxs) != 0:
        for j in idxs:
            point.append(list(line[j]))
    warn_point.append(point)

print(warn_point)

pred_mask = np.zeros([shape[2],shape[1],shape[0]])

for i in range(len(warn_point)):
    if len(warn_point[i]) != 0:
        for j in warn_point[i]:
            print(j)
            #pred_mask[j[2],j[1],j[0]] = 1
            pred(pred_mask,j)


label_path = '../heart_data_nii/label/0.nii'
ori_label = sitk.ReadImage(label_path)
out_data = (pred_mask > 0.5).astype(np.uint8)
#out_data = pred_mask
out_path = '2-prediction/warn_0.nii'
out_data = sitk.GetImageFromArray(out_data)
out_data.SetSpacing(ori_label.GetSpacing())
sitk.WriteImage(out_data, out_path)

print(pred_mask[123,161,183])

epi_img = nib.load('2-prediction/warn_0.nii')
print(epi_img.shape)
epi_img_data = epi_img.get_data()
print(epi_img_data[183,161,123])

slice_0 = epi_img_data[179, :, :]
slice_1 = epi_img_data[:, 160, :]
slice_2 = epi_img_data[:, :, 123]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
plt.show()
