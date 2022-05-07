import glob, os
import numpy as np

files = []
with open("stimuli_list.txt", "r") as f:
    files = f.readlines()
# print(files)

os.chdir("/scratch/CSAI/image_feature/InceptionV3/train")
# path = "/scratch/CSAI/image_feature/InceptionV3/train"
path = "."


# print(os.getcwd())

for layer in [248, 279, 310, -1]:
    print(layer)
    patht = path + "/" + str(layer)
    values = {}
    # values[layer] = {}
    # print(patht)
    # print(os.listdir(patht))

    file_list = os.listdir(patht)
    # print(file_list)
    # print(os.getcwd())
    for line in files:
        if "COCO" in line:
            # print(line)
            if (str(line.split("\n")[0])+".npy") in file_list:
                # print(line, line+".npy")
                id = int(line.split("\n")[0].split("_")[2].split(".")[0])
                # print(id)
                values[id] = np.load(str(layer)+"/"+line.split("\n")[0]+".npy", allow_pickle=True)
    print(len(values.keys()))
    # print(values[layer].keys())
    # for filename in os.listdir(patht):
    #     if(filename.slit("_")[0] == "COCO"):
    #         if ".".join(filename.split("/")[-1].split(".")[:-1]) in files:
    #             print(filename)
    #             id = int(filename.split("_")[2].split(".")[0])
    #             values[layer][id] = np.load(filename, allow_pickle=True)
# print(values.keys())
    np.save("./Required_"+str(layer)+".npy", values)