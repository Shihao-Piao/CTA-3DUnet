# Vessel Segmentation
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from net_model import UNet3D, EncoderUNet3D, DecoderUNet3D
from utils import NewDiceLoss, cal_dice_loss, MyDataset_UNet, check_and_create_path, Logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os
# 只使用一个线程
os.environ["OMP_NUM_THREADS"] = "1"



data_dir = '../heart_data_nii/train/'
label_dir = '../heart_data_nii/label/'
checkpoint_dir = 'checkpoints/test/'
model_save_dir = checkpoint_dir + 'save/'
check_and_create_path(model_save_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
OLD_EPOCH = 0
num_class = 1
label_list = [1]
sys.stdout = Logger(checkpoint_dir + 'log.txt')


def train_net(net, epochs=1000, batch_size=1, learning_rate=0.002):
    # Split dataset
    from sklearn.model_selection import train_test_split
    patient_list = os.listdir(data_dir)
    patient_list.sort()
    train_list_initail, test_list = train_test_split(patient_list[:], test_size=0.2, random_state=12)
    train_list, val_list = train_test_split(train_list_initail, test_size=0.25, random_state=12)
    print(len(train_list), len(val_list), len(test_list), train_list, val_list, test_list)

    train_set = MyDataset_UNet(train_list, data_dir, label_dir, valid_flag=False)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_set = MyDataset_UNet(val_list, data_dir, label_dir, valid_flag=True)
    valid_loader = data.DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1)

    # optimizer = torch.opim.SGD([{'params': net.parameters()}], lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=learning_rate, weight_decay=0.0005)
    # criterion = nn.BCELoss()
    criterion = NewDiceLoss()

    print('''Starting training:
            Epochs:{}
            Batch size:{}
            Learning rate:{}
            Training size:{}
            Validation size:{}
            '''.format(epochs, batch_size, learning_rate, len(train_list), len(val_list)))

    best_val_loss = np.inf
    best_model_path = 'checkpoints/best_model-'
    bad_epoch = 0
    for epoch in range(OLD_EPOCH, OLD_EPOCH + epochs):
        time1 = time.time()

        # Train
        net.train()
        # Set learning rate
        #training_lr = learning_rate * (0.7 ** ((epoch - 0) // 30))
        training_lr = learning_rate * (0.7 ** ((epoch - 0) // 50))
        for param_group in optimizer.param_groups:
            param_group['lr'] = training_lr
        train_loss = np.zeros([num_class])
        count = 0

        for index, (img, label) in enumerate(train_loader):

            if epoch == 1:
                print(img.shape)

            #img = img.cuda()
            img = img.to(device)

            #label = label.cuda()
            label = label.to(device)

            out = net(img)
            if epoch == 1:
                print(img.shape)

            loss = criterion(out, label, beta=3)
            train_loss[0] += loss.item()
            count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(count,train_loss)
            del loss, img, label, out
        train_loss = train_loss / count
        print("train done")

        # Validation
        net.eval()
        val_loss = np.zeros([num_class])
        count = 0
        for index, (img, label) in enumerate(valid_loader):
            # patch version 2
            z_cut_size = 32  # 160-16  144-32  128-48  112-64  96-80  80-96  64-32  48-32
            z_step = 16
            patch_number = math.ceil((img.shape[2] - z_cut_size) / z_step) + 1
            out = np.zeros(img.shape[2:])
            pred_count = np.zeros(img.shape[2:])
            for i in range(patch_number):
                start_point = i * z_step
                end_point = min(i * z_step + z_cut_size, img.shape[2])
                # print(start_point, end_point)
                img_patch = img[:, :, start_point:end_point, :, :]
                img_patch.requires_grad_(requires_grad=False)
                #img_patch = img_patch.cuda()
                img_patch = img_patch.to(device)

                out_patch = net(img_patch)
                out_patch = out_patch.squeeze().cpu().detach().numpy()
                out[start_point:end_point, :, :] += out_patch
                pred_count[start_point:end_point, :, :] += 1
                del img_patch, out_patch
            out = out / pred_count

            if np.max(out) > 1:
                print('11111111111111111111111111111111111')
            out = (out > 0.5).astype(np.uint8)
            label = label.squeeze().cpu().detach().numpy()
            val_loss[0] += cal_dice_loss(out, label, beta=3)
            count += 1
            del img, label, out
        val_loss = val_loss / count
        time_cost = time.time() - time1

        # Print epoch loss
        np.set_printoptions(precision=4)
        print('Epoch {0}/{1} : Loss:{2:.4f}, Valid loss: {3:.4f}, lr:{4:.6f}'
              .format(epoch + 1, epochs + OLD_EPOCH, np.mean(train_loss), np.mean(val_loss), training_lr), 1 - val_loss, time_cost)

        # Save the model
        val_loss = np.mean(val_loss)
        if np.mean(train_loss) * 0.2 < val_loss < best_val_loss:
            if os.path.exists(best_model_path) and val_loss > 0.28:
                os.remove(best_model_path)
            best_model_path = checkpoint_dir + 'best_model-' + '{0:.4f}.pth'.format(1 - val_loss)
            torch.save(net.state_dict(), best_model_path)
            best_val_loss = val_loss
            bad_epoch = 0
        else:
            bad_epoch += 1
            if bad_epoch == 60:
                print('******EXIT******: No improvement for 60 epochs!')
                sys.exit(0)
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), model_save_dir + '{}.pth'.format(epoch + 1))
            print('Model {} saved.'.format(epoch + 1))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    unet3d = UNet3D(input_channels=1, n_classes=1, f_channel=8)
    # net_encoder = EncoderUNet3D(channel_list=[1, 4, 8, 16, 32, 32])
    # net_decoder = DecoderUNet3D(channel_list=[32, 32, 16, 8, 4, num_class])

    #unet3d = unet3d.cuda()
    unet3d = unet3d.to(device)

    model_save_path = model_save_dir + str(OLD_EPOCH) + '.pth'
    if os.path.exists(model_save_path):
        unet3d.load_state_dict(torch.load(model_save_path))
        print('Model loaded from {}.'.format(model_save_path))
    else:
        print('Building a new model...')

    train_net(unet3d)
