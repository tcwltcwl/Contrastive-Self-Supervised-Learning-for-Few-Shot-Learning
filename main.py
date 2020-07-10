from data_util import augment_image
from data import ImageDataset
from model import construct_encoder
from model_util import get_optimizer, loss_fn

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

image_dir = os.path.dirname(__file__) + '/images'
dataset = ImageDataset(root_dir = image_dir,
                       num_augments = 5,
                       transform = augment_image())

train_dataset = DataLoader(dataset=dataset,
                           batch_size=1,
                           shuffle=True,
                           num_workers=0)

use_gpu = torch.cuda.is_available()

def visualize(images, num_images):
    
    plt.figure()
    
    for i in range(num_images):
        plt.subplot(np.sqrt(num_images),np.sqrt(num_images),i+1)
        plt.imshow(images[0][i])

    
def train(epochs, encoder, optimizer, criterion):
    
    train_losses = []
    valid_losses = []
    
    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        
        for phase in ['train']:
            
            if phase == 'train':
                encoder.train()
            elif phase == 'val':
                encoder.eval()
            
            for i, images in enumerate(train_dataset):
                
                if use_gpu:
                    images = Variable(images.squeeze()).cuda()
                else:
                    images = Variable(images.squeeze())
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    encoder_output = encoder(images).squeeze()
                    loss = criterion(encoder_output, trade_off = 0.5)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        
                    elif phase == 'val':
                        valid_loss += loss.item()
            
#        val_loss = val(epoch, encoder)
#        
        train_losses.append(train_loss)
#        valid_losses.append(val_loss)
            
#            visualize(images, 9)
            
    return train_losses

if __name__ == "__main__":
        
    encoder = construct_encoder()
    optimizer = get_optimizer(encoder, 0.005)
    criterion = loss_fn()
    
    train_losses = train(10, encoder, optimizer, criterion)