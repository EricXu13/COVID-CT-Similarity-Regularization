import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


from PIL import Image
from skimage import io
from skimage.color import gray2rgb


class COVIDxDataset(Dataset):
    def __init__(self, data_dir, split_file, transform=None, im_size=256):
        data_dir = os.path.expanduser(data_dir)
        self.data_dir = os.path.join(data_dir, '2A_images')
        split_file = os.path.join(data_dir, split_file)
        self.files, self.classes = self._get_files(split_file)
        self.count = len(self.classes)
        
        self.transform = transform
        
    def __len__(self):
        return self.count
    
    def _get_files(self, split_file):
        files, classes = [], []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                fname, cls = line.strip('\n').split()[:2]
                files.append(os.path.join(self.data_dir, fname))
                classes.append(int(cls))
        
        files = np.asarray(files)
        classes = torch.LongTensor(classes)
        return files, classes

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path, img_cls = self.files[idx], self.classes[idx]
#         img = gray2rgb(io.imread(img_path))
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)            

        return img, img_cls
    
    
class ExternalDataset(Dataset):
    def __init__(self, data_dir, transform=None, im_size=256):
        self.data_dir = os.path.expanduser(data_dir)
        self.files, self.classes = self._get_files()
        self.count = len(self.classes)
        self.transform = transform
        
    def __len__(self):
        return self.count
        
    def _get_files(self, ):
        file_names = ['non-COVID', 'COVID']
        files, classes = [], []
        for i in range(2):
            class_dir = os.path.join(self.data_dir, file_names[i])
            for fname in os.listdir(class_dir):
                files.append(os.path.join(class_dir, fname))
                classes.append(int(i))
        files = np.asarray(files)
        classes = torch.LongTensor(classes)
        return files, classes

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path, img_cls = self.files[idx], self.classes[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)            

        return img, img_cls