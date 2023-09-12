import nrrd
import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk

# nrrd 文件保存路径
data_path=r'../heart_data/Train/'
save_path='../heart_data_nii/train/'

'''
files = os.listdir(data_path)
files.sort()
'''

def nrrd2nii():


    for i in range(40):

        print(i)

        data_path = r'../heart_data/Train/'+str(i)+'.nrrd'
        save_path = '../heart_data_nii/train/'+str(i)+'.nii'

        data,options=nrrd.read(data_path)  # 读取 nrrd 文件
        img=nib.Nifti1Image(data,np.eye(4)) # 将 nrrd 文件转换为 .nii 文件
        nib.save(img,save_path) # 保存 nii 文件

        data_path = r'../heart_data/Train_Masks/'+str(i)+'.nrrd'
        save_path = '../heart_data_nii/label/'+str(i)+'.nii'

        data,options=nrrd.read(data_path)  # 读取 nrrd 文件
        img=nib.Nifti1Image(data,np.eye(4)) # 将 nrrd 文件转换为 .nii 文件
        nib.save(img,save_path) # 保存 nii 文件

def nii2mha(data_path,out_path):
    data = sitk.ReadImage(data_path)
    # data = read_dicom_data(data_path)
    data = sitk.GetArrayFromImage(data)
    img = sitk.GetImageFromArray(data)
    sitk.WriteImage(img,out_path)

data = '2-prediction/2.nii'
out = '2-prediction/2.mha'
nii2mha(data,out)