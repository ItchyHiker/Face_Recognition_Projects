import os, sys

import cv2
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CASIAWebFace(Dataset):
    def __init__(self, root_path, file_list, transform=None):
        self.root_path = root_path
        self.transform = transform
        
        image_list = []
        label_list = []
        with open(file_list) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_path, label_name = line.split(' ')
                image_list.append(image_path)
                label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.num_class = len(np.unique(self.label_list))
        print("CASIA dataset size:", len(self.image_list), '/', self.num_class)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = self.label_list[index]

        image = cv2.imread(os.path.join(self.root_path, image_path), 1) # Read BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        flip = np.random.choice(2) * 2 - 1
        if flip == 1:
            image = cv2.flip(image, 1)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)

        return image, label

    def __len__(self):
        return len(self.image_list)

class SiameseCASIAWebFace(Dataset):
    def __init__(self, root_path, file_list, transform=None):
        self.root_path = root_path
        self.transform = transform
        
        image_list = []
        label_list = []
        with open(file_list) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_path, label_name = line.split(' ')
                image_list.append(image_path)
                label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.label_set = set(self.label_list)
        self.num_class = len(self.label_set)
        self.label_array = np.array(label_list)
        self.label_to_idxs = {label:np.where(self.label_array==label)[0]
                                for label in self.label_set}
        print("CASIA dataset size:", len(self.image_list), '/', self.num_class)
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        imageL_path, labelL = self.image_list[index], self.label_list[index]
        if target == 1: # 
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_idxs[labelL])
                
        else:
            siamese_label = np.random.choice(list(self.label_set - set([labelL])))
            siamese_index = np.random.choice(self.label_to_idxs[siamese_label])
        imageR_path = self.image_list[siamese_index]
        labelR = self.label_list[siamese_index]
        
        imageL = cv2.imread(os.path.join(self.root_path, imageL_path), 1)
        imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)
        imageR = cv2.imread(os.path.join(self.root_path, imageR_path), 1)
        imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB)
        
        flip = np.random.choice(2) * 2 - 1
        if flip == 1:
            imageL = cv2.flip(imageL, 1)
            imageR = cv2.flip(imageR, 1)
        if self.transform is not None:
            imageL = self.transform(imageL)
            imageR = self.transform(imageR)
        else:
            imageL = torch.from_numpy(imageL)
            imageR = torch.from_numpy(imageR)
        
        return imageL, imageR, target
            

class LFW(Dataset):
    def __init__(self, root_path, file_list, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.nameLs = []
        self.nameRs = []
        self.flags = []
        self.folds = []

        with open(file_list) as f:
            lines = f.readlines()
            for i, line in enumerate(lines[1:]):
                line = line.strip()
                p = line.split('\t')
                fold = i // 600 # 6000 pairs, so there should be 10 folds
                if len(p) == 3: # same person
                    nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                    nameR = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
                    flag = 1
                elif len(p) == 4:
                    nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                    nameR = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
                    flag = -1
                self.nameLs.append(nameL)
                self.nameRs.append(nameR)
                self.flags.append(flag)
                self.folds.append(fold)

    def __getitem__(self, index):
        img_l = cv2.imread((os.path.join(self.root_path, self.nameLs[index])), 1)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r = cv2.imread((os.path.join(self.root_path, self.nameRs[index])), 1)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img_l = self.transform(img_l)
            img_r = self.transform(img_r)
        else:
            img_l = torch.from_numpy(img_l)
            img_r = torch.from_numpy(img_r)
        return {'pair': [img_l, img_r], 
                'label': self.flags[index], 
                'fold': self.folds[index]}

    def __len__(self):
        return len(self.flags)

if __name__ == "__main__":
    train_data = SiameseCASIAWebFace('./training_data/CASIA', 'annotations/CASIA_anno.txt')
