import os
import glob
import random
import shutil

dataset_path = "/Users/Pey/Downloads/madg/madg/train"
output_path = "/Users/Pey/Downloads/madg/madg/valid"

output_sat_path = os.path.join(output_path, "sat")
output_map_path = os.path.join(output_path, "map")
dataset_sat_path = os.path.join(dataset_path, "sat")
dataset_map_path = os.path.join(dataset_path, "map")

if not os.path.exists(output_sat_path):
    os.makedirs(output_sat_path)
if not os.path.exists(output_map_path):
    os.makedirs(output_map_path)

if not os.path.exists(dataset_path):
    print(f"dataset_path: {dataset_path} does not exists!")

if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset_sat_glob = glob.glob(f"{dataset_path}/sat/*sat.jpg") + glob.glob(f"{dataset_path}/sat/*.png")
dataset_map_glob = glob.glob(f"{dataset_path}/map/*sat.jpg") + glob.glob(f"{dataset_path}/map/*.png")

dataset_sat_files = {i:file for i, file in enumerate(dataset_sat_glob)}
dataset_mask_files = {}
for index, file in dataset_sat_files.items():
    filename_with_ext = os.path.basename(file)
    filename, filename_ext = os.path.splitext(filename_with_ext)
    mapfile = os.path.join(dataset_path, "map", f"{filename.replace('sat', 'mask')}.png")
    # print(f"is {file} in {os.path.join(dataset_path, 'map')}? ")
    assert mapfile in dataset_map_glob
    dataset_mask_files[index] = mapfile

validset_indices = set()

print(f"dataset_path: {dataset_path}")
print(f"# of dataset_sat_files: {len(dataset_sat_files)}")
print(f"# of dataset_mask_files: {len(dataset_mask_files)}")
print(f"dataset_sat_files[0]: {dataset_sat_files[0]}")
print(f"dataset_mask_files[0]: {dataset_mask_files[0]}")
print(f"output_path: {output_path}")
input(f"Press enter to continue: ")

P = int(input(f"Enter the percentage of validation set (out of 100): "))
N_total = len(dataset_sat_files)

N_valid = int(P/100*N_total)
print(f"N_valid total: {N_valid}")

i = 0
while i < N_valid:
    rand_index = random.randint(0, N_total-1)
    if rand_index not in validset_indices:
        validset_indices.add(rand_index)
        i += 1

print(f"Result:\n"
      f"# of train points: {len(dataset_sat_files)}\n"
      f"# of valid points: {len(validset_indices)}")

input(f"Press enter to write validset to: {output_path}")

everyN = N_valid // 10
for index in validset_indices:

    satfile = dataset_sat_files[index]
    mapfile = dataset_mask_files[index]

    shutil.move(satfile, output_sat_path)
    shutil.move(mapfile, output_map_path)
    if index % everyN == 0:
        print(f"[File Move: {index+1}] moved {satfile} to {output_sat_path}")
        print(f"[File Move: {index+1}] moved {mapfile} to {output_map_path}\n")

print(f"# of files in: dataset_sat_path: {len(os.listdir(dataset_sat_path))}")
print(f"# of files in: output_sat_path:  {len(os.listdir(output_sat_path))}")
print(f"# of files in: dataset_map_path: {len(os.listdir(dataset_map_path))}")
print(f"# of files in: output_map_path:  {len(os.listdir(output_map_path))}")
print(f"Finished!")

