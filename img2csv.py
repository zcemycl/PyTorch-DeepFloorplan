import csv
import cv2
import matplotlib.pyplot as plt
import tqdm
import sys
sys.path.append('./utils/')
from rgb_ind_convertor import *
from util import *

paths = open('./dataset/r3d_test.txt','r').read().splitlines()
image_paths = [p.split('\t')[0] for p in paths] # image 
gt1 = [p.split('\t')[1] for p in paths] # 1 wall
gt2 = [p.split('\t')[2] for p in paths] # 2 window,door
gt3 = [p.split('\t')[3] for p in paths] # 3 rooms
gt4 = [p.split('\t')[-1] for p in paths] # close wall

with open('r3d2.csv','w',newline='') as csvfile:
    fieldnames = ['image','boundary','room','door']
    writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()
    for idx,image_path in enumerate(tqdm.tqdm(image_paths)):
        img = cv2.imread(image_path[1:])
        wall = cv2.imread(gt1[idx][1:],0)
        door = cv2.imread(gt2[idx][1:],0)
        room = rgb2ind(cv2.imread(gt3[idx][1:])[:,:,::-1],
                color_map=floorplan_fuse_map)
        boundary = np.zeros(door.shape)
        boundary[door>0] = 1
        boundary[wall>0] = 2
        image = cv2.resize(img,(512,512)).flatten().astype(np.uint8)
        boundary = cv2.resize(boundary,(512,512)).flatten().astype(np.uint8)
        room = cv2.resize(room,(512,512)).flatten().astype(np.uint8)
        door = cv2.resize(door,(512,512)).flatten().astype(np.uint8)

        writer.writerow(
        {'image':list(image),'boundary':list(boundary),
         'room':list(room),'door':list(door)})
    


