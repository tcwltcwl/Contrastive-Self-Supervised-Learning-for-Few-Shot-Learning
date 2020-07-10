import torch.utils.data as data
from PIL import Image
import torch
import os

class ImageDataset(data.Dataset):
    
    def __init__(self, root_dir, num_augments=2, transform=None):
        
        self.root_dir = root_dir
        self.img_names = os.listdir(root_dir)
        self.num_augments = num_augments
        self.transform = transform
        
    def __getitem__(self, index):
        
        output = []
        img = Image.open(self.root_dir + '/' + self.img_names[index]).convert('RGB')
        
#        if self.transform is not None:
#            img = self.transform(img)
#            
#        return img, index
            
        for i in range(self.num_augments):
            if self.transform is not None:
                img_transform = self.transform(img)
                
#            img_transform = img_transform.permute(1,2,0)
            output.append(img_transform)
            
        output = torch.stack(output, axis=0)
            
        return output
            
    def __len__(self):
        
        return len(self.img_names)
