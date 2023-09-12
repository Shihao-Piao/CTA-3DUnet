import os
import sys
import numpy as np
import torch
from torch.utils import data
import math
import random
import SimpleITK as sitk
from scipy import ndimage
from skimage import transform

label_list = [1]
E_Z1 = 4
E_Y1 = 8
E_X1 = 8


def summary(total_tp, total_fn, total_fp):
    # 按病灶算
    print('\nTotal tp:', np.sum(total_tp), 'Total fn:', np.sum(total_fn), 'Total fp:', np.sum(total_fp))
    print('Recall:', np.sum(total_tp) / (np.sum(total_tp) + np.sum(total_fn)), 'Precision:', np.sum(total_tp) / (np.sum(total_tp) + np.sum(total_fp)))
    # 按人头算平均值
    print('Mean tp', np.mean(total_tp), 'Std tp', np.std(total_tp))
    print('Mean fn', np.mean(total_fn), 'Std fn', np.std(total_fn))
    print('Mean fp', np.mean(total_fp), 'Std fp', np.std(total_fp))
    recall = total_tp / (total_tp + total_fn)
    precision = total_tp / (total_tp + total_fp)
    print('Mean recall', np.mean(recall), 'Std recall', np.std(recall))
    print('Mean precision', np.mean(precision), 'Std precision', np.std(precision))


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def read_dicom_data(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def load_data(data_path):
    data = sitk.ReadImage(data_path)
    # data = read_dicom_data(data_path)
    data = sitk.GetArrayFromImage(data)
    # data = np.clip(data, -100, 1000)  # -200, 600    -100, 300   -150, 400
    # data = data[13:-14, 40:-40, 24:-24]
    # data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data - np.mean(data)) / np.std(data)
    # data = (data != 0).astype(np.float32) * (data - np.mean(data)) / np.std(data)
    # data_temp = data.ravel()[np.flatnonzero(data)]
    # data = (data != 0).astype(np.float32) * (data - np.mean(data_temp)) / np.std(data_temp)
    # data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data


def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MyDataset_UNet(data.Dataset):
    def __init__(self, instance_list, data_dir, label_dir, valid_flag=False):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.instance_list = instance_list
        self.valid_flag = valid_flag

    def __getitem__(self, index):
        '''
        instance_name = self.instance_list[index]
        data_path = self.data_dir + instance_name #+ '/'
        data_path_1 = data_path# + 'image_c.nii.gz'  # zengqiang
        data_path_2 = data_path# + 'image_p.nii.gz'
        label_path = self.label_dir + instance_name #+ '/label.nii.gz'
        # print(data_path_1)

        # load the data and label
        data_1 = load_data(data_path_1).astype(np.float32)
        #print(data_1.shape)
        data_2 = load_data(data_path_2).astype(np.float32)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label).astype(np.float32)

        # padding
        padding_z = (16 - data_1.shape[0] % 16) % 16
        data_1 = np.pad(data_1, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        data_2 = np.pad(data_2, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        label = np.pad(label, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        padding_y = (16 - data_1.shape[1] % 16) % 16
        data_1 = np.pad(data_1, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        data_2 = np.pad(data_2, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        label = np.pad(label, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        padding_x = (16 - data_1.shape[2] % 16) % 16
        data_1 = np.pad(data_1, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        data_2 = np.pad(data_2, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        label = np.pad(label, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        #print(data_1.shape, label.shape)
        # Data augmentation for training
        if not self.valid_flag:
            # random crop to same size
            z_size, y_size, x_size = data_1.shape
            z_crop, y_crop, x_crop = (32, 512, 512)  # (160, 320, 320)
            z_random = np.random.randint(0, z_size - z_crop + 1)
            y_random = np.random.randint(0, y_size - y_crop + 1)
            x_random = np.random.randint(0, x_size - x_crop + 1)
            data_1 = data_1[z_random:z_random + z_crop, y_random:y_random + y_crop, x_random:x_random + x_crop]
            data_2 = data_2[z_random:z_random + z_crop, y_random:y_random + y_crop, x_random:x_random + x_crop]
            label = label[z_random:z_random + z_crop, y_random:y_random + y_crop, x_random:x_random + x_crop]

            # if random.randint(0, 1):
            #     rotate_degree = random.uniform(-5, 5)
            #     data = transform.rotate(data.transpose(1, 2, 0), angle=rotate_degree, resize=False, cval=-1)
            #     data = data.transpose(2, 0, 1)
            #     label = transform.rotate(label.transpose(1, 2, 0), angle=rotate_degree, resize=False, cval=0)
            #     label = label.transpose(2, 0, 1)
            # if random.randint(0, 1):
            #     data = np.flip(data, axis=0)
            #     label = np.flip(label, axis=0)
            # if random.randint(0, 1):
            #     data = np.flip(data, axis=1)
            #     label = np.flip(label, axis=1)
            # if random.randint(0, 1):
            #     data = np.flip(data, axis=2)
            #     label = np.flip(label, axis=2)
            # if random.randint(0, 1):
            #     shift_value = random.uniform(-0.1, 0.1)
            #     scale_value = random.uniform(0.9, 1.1)
            #     # print(shift_value, scale_value, '666666')
            #     data = data * scale_value + shift_value

        data_1 = data_1[np.newaxis, :, :, :].astype(np.float32)
        data_2 = data_2[np.newaxis, :, :, :].astype(np.float32)
        data = np.concatenate((data_1, data_2), axis=0)

        label = (label > 0)[np.newaxis, :, :, :].astype(np.float32)
        # print(data.shape, label.shape)

        # return data[:, :, 64:-64, 100:356], label[:, :, 64:-64, 100:356]
        return data, label
        '''
        instance_name = self.instance_list[index]
        data_path = self.data_dir + instance_name  # + '/'
        label_path = self.label_dir + instance_name  # + '/label.nii.gz'
        # print(data_path_1)

        # load the data and label
        data_1 = load_data(data_path).astype(np.float32)
        # print(data_1.shape)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label).astype(np.float32)

        # padding
        padding_z = (16 - data_1.shape[0] % 16) % 16
        data_1 = np.pad(data_1, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        label = np.pad(label, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')

        padding_y = (16 - data_1.shape[1] % 16) % 16
        data_1 = np.pad(data_1, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        label = np.pad(label, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')

        padding_x = (16 - data_1.shape[2] % 16) % 16
        data_1 = np.pad(data_1, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        label = np.pad(label, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        #print(data_1.shape, label.shape)
        # Data augmentation for training
        if not self.valid_flag:
            # random crop to same size
            z_size, y_size, x_size = data_1.shape
            z_crop, y_crop, x_crop = (64, 512, 512)  # (160, 320, 320)
            z_random = np.random.randint(0, z_size - z_crop + 1)
            y_random = np.random.randint(0, y_size - y_crop + 1)
            x_random = np.random.randint(0, x_size - x_crop + 1)

            data_1 = data_1[z_random:z_random + z_crop, y_random:y_random + y_crop, x_random:x_random + x_crop]
            label = label[z_random:z_random + z_crop, y_random:y_random + y_crop, x_random:x_random + x_crop]


        data_1 = data_1[np.newaxis, :, :, :].astype(np.float32)

        data = data_1

        label = (label > 0)[np.newaxis, :, :, :].astype(np.float32)
        # print(data.shape, label.shape)

        # return data[:, :, 64:-64, 100:356], label[:, :, 64:-64, 100:356]
        return data, label

    def __len__(self):
        return len(self.instance_list)


if __name__ == '__main__':

    np.set_printoptions(precision=3)
    aa = np.array([1, 2, 3, 4.456789, 5, 6])
    print(aa)

    data_dir = '../../Data/Patients_image_352/'
    label_dir = '../../Data/Patients_label_352/'
    # Extract every fold from total list
    patient_list = os.listdir(data_dir)
    train_list = []
    val_list = []
    for index in range(len(patient_list)):
        if index % 2 == 0:
            if index < 12:
                train_list.append(patient_list[index])
            else:
                val_list.append(patient_list[index])
    print(len(train_list), len(val_list), train_list, val_list)
    train_set = MyDataset(train_list, data_dir, label_dir, valid_flag=True)
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    for iter_index, (batch_data, batch_label) in enumerate(train_loader):
        print(iter_index)
        print('data shape:', batch_data.shape)
        print('label shape:', batch_label.shape)
