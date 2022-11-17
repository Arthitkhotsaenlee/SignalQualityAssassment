import os




datafolder_path = "Z:\sample_data\quality_assessment"
folder_list = os.listdir(datafolder_path)
print(folder_list)
for i in folder_list:
    file_path = os.path.join(datafolder_path,i)
    file_list = os.listdir(file_path)
    print(file_list)

