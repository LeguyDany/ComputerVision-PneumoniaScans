import os
import shutil

input_path = input("Veuillez entrer le chemin d'accès jusqu'au fichier chest_Xray sans l'inclure : ")

dataset_prefix = input_path + '/chest_Xray'

prefix = input_path + '/better_repartition'

train_path = prefix + '/TRAIN'
val_path = prefix + '/VALIDATION'
test_path = prefix + '/TEST'

os.mkdir(prefix)
os.mkdir(prefix + "/ALL")
os.mkdir(train_path)
os.mkdir(val_path)
os.mkdir(test_path)

for folder in os.listdir(dataset_prefix + ""):
    if folder == '.DS_Store': continue 

    for inner_folder in os.listdir(f"{dataset_prefix}/{folder}"):
        if inner_folder == '.DS_Store': continue 

        for file in os.listdir(f"{dataset_prefix}/{folder}/{inner_folder}"):
            if file == '.DS_Store': continue 

            shutil.copyfile(f"{dataset_prefix}/{folder}/{inner_folder}/{file}", f"{prefix}/ALL/{file}")
            
dataset_size = len(os.listdir(prefix + '/ALL'))
train_size = int(dataset_size * 0.8)
val_size = int(dataset_size * 0.1) + 1
test_size = int(dataset_size * 0.1) + 1

for i, filename in enumerate(os.listdir(prefix + '/ALL')):
    if i < train_size:
        shutil.copyfile(f'{prefix}/ALL/{filename}', f'{train_path}/{filename}')
    elif i < train_size + val_size:
        shutil.copyfile(f'{prefix}/ALL/{filename}', f'{val_path}/{filename}')
    else:
        shutil.copyfile(f'{prefix}/ALL/{filename}', f'{test_path}/{filename}')
