import os
import glob


def delete_files(directory):
    # 遍历文件夹及其所有子文件夹
    for root, dirs, files in os.walk(directory):
        # 使用glob找到当前文件夹中所有符合特定模式的文件路径
        # 模式 'ROI_*_t1.nii' 能够正确匹配例如 'ROI_Brats17_2013_3_1_t1.nii'
        pattern = os.path.join(root, "ROI_*_t1.nii")
        matched_files = glob.glob(pattern)

        if matched_files:
            print(f"Matched files in {root}: {matched_files}")
        else:
            print(f"No matched files in {root}")

        for filepath in matched_files:
            # 检查文件是否确实存在
            if os.path.isfile(filepath):
                print(f"Deleting: {filepath}")
                os.remove(filepath)  # 删除文件
            else:
                print(f"File not found: {filepath}")


# 调用函数，'your_directory_path' 替换为你的目标文件夹路径
delete_files(r"D:\Data\BraTs2017\Data")
