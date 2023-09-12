# Test for segmentation
import os
import sys
import time
import math
import numpy as np
import torch
import SimpleITK as sitk
from net_model import UNet3D
from utils import check_and_create_path, load_data, Logger
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')



data_dir = '../heart_data_nii/train/'
label_dir = '../heart_data_nii/label/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = 'checkpoints/test/backup/f8-0.002-500/best_model-0.7459.pth'
result_dir = '2-prediction/'
check_and_create_path(result_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
num_class = 1
label_list = [1]
sys.stdout = Logger('2-prediction/predict_log.txt')


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    patient_list = os.listdir(data_dir)
    patient_list.sort()
    train_list_initail, test_list = train_test_split(patient_list[:], test_size=0.2, random_state=12)
    train_list, val_list = train_test_split(train_list_initail, test_size=0.25, random_state=12)
    test_list = patient_list
    print('Test on {} samples: {}'.format(len(test_list), test_list))

    #net = UNet3D(input_channels=2, n_classes=1, f_channel=8)
    net = UNet3D(input_channels=1, n_classes=1, f_channel=8)
    net = net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    total_dice = 0
    total_acc = 0
    total_sen = 0
    total_pre = 0
    time000 = time.time()
    for test_file in test_list:
        print('Predicting', test_file)
        time_s = time.time()
        data_path = data_dir + test_file# + '/'
        # data_path_1 = data_path + os.listdir(data_path)[0]  # + '/'  # zengqiang
        # data_path_2 = data_path + os.listdir(data_path)[1]  # + '/'
        data_path_1 = data_path #+ 'image_c.nii.gz'  # zengqiang
        #data_path_2 = data_path# + 'image_p.nii.gz'
        label_path = label_dir + test_file #+ '/label.nii.gz'

        # load the data and label
        data_1 = load_data(data_path_1).astype(np.float32)
        #data_2 = load_data(data_path_2).astype(np.float32)
        ori_label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(ori_label)

        # padding
        padding_z = (16 - data_1.shape[0] % 16) % 16
        data_1 = np.pad(data_1, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        #data_2 = np.pad(data_2, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        # label = np.pad(label, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        padding_y = (16 - data_1.shape[1] % 16) % 16
        data_1 = np.pad(data_1, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        #data_2 = np.pad(data_2, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        # label = np.pad(label, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        padding_x = (16 - data_1.shape[2] % 16) % 16
        data_1 = np.pad(data_1, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        #data_2 = np.pad(data_2, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        # label = np.pad(label, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')

        data_1 = data_1[np.newaxis, np.newaxis, :, :, :].astype(np.float32)
        #data_2 = data_2[np.newaxis, np.newaxis, :, :, :].astype(np.float32)
        data = np.concatenate((data_1, data_1), axis=1)

        data = data_1

        label = (label > 0).astype(np.float32)
        data = torch.from_numpy(data)

        test_mode = 2

        # original version 0
        if test_mode == 0:
            data = data.cuda()
            out = net(data)
            pred_mask = out.squeeze().cpu().detach().numpy()

        # patch version 1
        elif test_mode == 1:
            cut_number = 12
            z_cut_size = data.shape[2] // cut_number
            pred_mask = np.zeros(data.shape[2:])
            for i in range(cut_number):
                start_point = max(0, i * z_cut_size - 10)
                end_point = min(data.shape[2], (i + 1) * z_cut_size + 10)
                # print(start_point, end_point)
                img_patch = data[:, :, start_point:end_point, :, :]
                img_patch.requires_grad_(requires_grad=False)
                img_patch = img_patch.cuda()

                out_patch = net(img_patch)
                out_patch = out_patch.squeeze().cpu().detach().numpy()
                pred_mask[start_point:end_point, :, :] += out_patch
                del img_patch, out_patch
                if i > 0:
                    pred_mask[start_point:start_point + 20, :, :] /= 2

        # patch version 2
        elif test_mode == 2:
            z_cut_size = 128
            z_step = 32
            patch_number = math.ceil((data.shape[2] - z_cut_size) / z_step) + 1
            pred_mask = np.zeros(data.shape[2:])
            for i in range(patch_number):
                start_point = i * z_step
                end_point = min(i * z_step + z_cut_size, data.shape[2])
                # print(start_point, end_point)
                img_patch = data[:, :, start_point:end_point, :, :]
                img_patch.requires_grad_(requires_grad=False)
                img_patch = img_patch.to(device)

                out_patch = net(img_patch)
                out_patch = out_patch.squeeze().cpu().detach().numpy()
                pred_mask[start_point:end_point, :, :] += out_patch
                del img_patch, out_patch
                if i > 0:
                    pred_mask[start_point:start_point + 96, :, :] /= 2

        # patch version 3
        elif test_mode == 3:
            cut_size = np.array([160, 320, 320])
            cut_step = np.array([144, 304, 304])
            patch_number = np.ceil((np.array(data.shape[2:]) - cut_size) / cut_step).astype(np.int) + 1
            pred_mask = np.zeros(data.shape[2:])
            pred_count = np.zeros(data.shape[2:])
            for i_z in range(patch_number[0]):
                for i_y in range(patch_number[1]):
                    for i_x in range(patch_number[2]):
                        z_start = i_z * cut_step[0]
                        z_end = min(i_z * cut_step[0] + cut_size[0], data.shape[2])
                        y_start = i_y * cut_step[1]
                        y_end = min(i_y * cut_step[1] + cut_size[1], data.shape[3])
                        x_start = i_x * cut_step[2]
                        x_end = min(i_x * cut_step[2] + cut_size[2], data.shape[4])
                        # print(start_point, end_point)
                        img_patch = data[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                        img_patch.requires_grad_(requires_grad=False)
                        img_patch = img_patch.cuda()

                        out_patch = net(img_patch)
                        out_patch = out_patch.squeeze().cpu().detach().numpy()
                        pred_mask[z_start:z_end, y_start:y_end, x_start:x_end] += out_patch
                        pred_count[z_start:z_end, y_start:y_end, x_start:x_end] += 1
                        del img_patch, out_patch
            pred_mask = pred_mask / pred_count

        if np.max(pred_mask) > 1:
            print('11111111111111111111111111111111111')

        out_data = (pred_mask > 0.5).astype(np.uint8)
        out_data = out_data[padding_z // 2: out_data.shape[0] - (padding_z - padding_z // 2),
                     padding_y // 2: out_data.shape[1] - (padding_y - padding_y // 2),
                     padding_x // 2: out_data.shape[2] - (padding_x - padding_x // 2)]
        out_path = result_dir + test_file  #+ '.nii.gz'
        out_data = sitk.GetImageFromArray(out_data)
        out_data.SetSpacing(ori_label.GetSpacing())
        sitk.WriteImage(out_data, out_path)

        pred_label = (pred_mask > 0.5).astype(np.uint8)
        pred_label = pred_label[padding_z // 2: pred_label.shape[0] - (padding_z - padding_z // 2),
                     padding_y // 2: pred_label.shape[1] - (padding_y - padding_y // 2),
                     padding_x // 2: pred_label.shape[2] - (padding_x - padding_x // 2)]

        # Post processing
        from skimage import measure
        all_labels = measure.label(pred_label, background=0, connectivity=3)
        properties = measure.regionprops(all_labels)
        areas = np.array([prop.area for prop in properties])
        # print(' ', np.sum(label), np.max(measure.label(label, background=0)), np.max(all_labels))
        final_label = np.zeros(all_labels.shape)
        for i, area in enumerate(areas):
            if area > 10:  # np.sum(label) * 0.01:
                # print('   ', area, i)
                final_label += (all_labels == (i + 1)).astype(np.float32)
        pred_label = final_label

        tp = np.sum(label * pred_label)
        tn = np.sum((1 - label) * (1 - pred_label))
        fp = np.sum((1 - label) * pred_label)
        fn = np.sum(label * (1 - pred_label))
        dice = (2 * tp + 10e-4) / (2 * tp + fp + 1 * fn + 10e-4)
        acc = (tp + tn + 10e-4) / (tp + tn + fp + fn + 10e-4)
        sen = (tp + 10e-4) / (tp + fn + 10e-4)
        pre = (tp + 10e-4) / (tp + fp + 10e-4)
        print(' ', dice, acc, sen, pre, tp, fp, fn)
        total_dice += dice
        total_acc += acc
        total_sen += sen
        total_pre += pre

        # # out_data = pred_label
        # # out_path = result_dir + test_file[:-5] + '_pred.nii.gz'
        # out_data = label * pred_label + (1 - label) * pred_label * 2 + label * (1 - pred_label) * 3
        # out_path = result_dir + str(dice)[:6] + '-' + test_file + '.nii.gz'
        # out_data = sitk.GetImageFromArray(out_data)
        # out_data.SetSpacing(ori_label.GetSpacing())
        # sitk.WriteImage(out_data, out_path)
        # print('    ', time.time() - time_s)

    print('Dice:', total_dice / len(test_list))
    print('Acc:', total_acc / len(test_list))
    print('Sen:', total_sen / len(test_list))
    print('Spe:', total_pre / len(test_list))
    print('Average Time:', (time.time() - time000) / len(test_list))


