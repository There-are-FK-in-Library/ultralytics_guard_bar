# import os
# import shutil
#
# def get_folder_size(folder_path):
#     """计算文件夹的大小（以字节为单位）"""
#     total_size = 0
#     for dirpath, dirnames, filenames in os.walk(folder_path):
#         for file in filenames:
#             file_path = os.path.join(dirpath, file)
#             # 累加文件大小
#             total_size += os.path.getsize(file_path)
#     return total_size
#
# def delete_train_folders(root_folder):
#     # 遍历根文件夹
#     for dirpath, dirnames, filenames in os.walk(root_folder):
#         for dirname in dirnames:
#             folder_path = os.path.join(dirpath, dirname)
#             # 检查文件夹名是否包含 "val" 或 "train"
#             if "val" in dirname or "train" in dirname:
#                 if "train" in dirname:
#                     # 检查 results.csv 是否存在以及其大小
#                     results_file_path = os.path.join(folder_path, "results.csv")
#                     if not os.path.isfile(results_file_path) or os.path.getsize(results_file_path) < 10 * 1024:  # 小于 10KB
#                         # 删除文件夹及其内容
#                         shutil.rmtree(folder_path)
#                         print(f"Deleted folder: {folder_path} (No valid results.csv or it is smaller than 10KB)")
#                 else:  # 如果是 "val" 文件夹
#                     # 获取文件夹大小
#                     folder_size = get_folder_size(folder_path)
#                     # 检查大小是否不超过 100MB
#                     if folder_size <= 10 * 1024 * 1024:  # 100MB
#                         # 删除文件夹及其内容
#                         shutil.rmtree(folder_path)
#                         print(f"Deleted folder: {folder_path} (Size: {folder_size / (1024 * 1024):.2f} MB)")
#
# # 示例用法
# root_folder = r'D:\WJ\Pycharm_workspace\ultralytics_guard_bar\runs\classify'  # 替换为你的根文件夹路径
# delete_train_folders(root_folder)