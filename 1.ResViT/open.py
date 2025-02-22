import numpy as np
import SimpleITK as sitk
from MeDIT import Visualization
from matplotlib import pyplot as plt

min_v = -200
max_v = 200
volume  = r"D:\Data\BraTs2017\BRATS2017\Brats17_2013_31_1\Brats17_2013_31_1_t1.nii"
#segment = r"C:\Data\0.Study\MIP\ISICDM\liver\train\liver_12\liver_seg12.nii"
# = r"C:\Data\0.Study\MIP\ISICDM\liver\train\liver_12\liver_nid12.nii"
image = sitk.ReadImage(volume)
image_arr = sitk.GetArrayFromImage(image)#.transpose(1,2,0)
#label_origan = sitk.ReadImage(segment)
#label_origan_arr = sitk.GetArrayFromImage(label_origan).transpose(1,2,0)
#label_nid = sitk.ReadImage(nid)
#label_nid_arr = sitk.GetArrayFromImage(label_nid).transpose(1,2,0)
#  image_arr[image_arr<min_v] = min_v
#  image_arr[image_arr>max_v] = max_v
#lab_arr2 =label_origan_arr.astype(np.float32)
#lab_arr2=[label_origan_arr.astype(np.float32),label_nid_arr.astype(np.float32)]

Visualization.Imshow3DArray(image_arr)