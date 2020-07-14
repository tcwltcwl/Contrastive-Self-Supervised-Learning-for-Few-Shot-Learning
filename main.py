from data_util import augment_image
from data import ImageDataset
from model import construct_encoder
from model_util import get_optimizer, loss_fn

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import time
import matplotlib.pyplot as plt

image_dir = os.path.dirname(__file__) + '/images'
num_augments = 10
batch_size = 5

train_dataset = ImageDataset(root_dir = image_dir,
                             num_augments = num_augments,
                             phase = 'train',
                             transform = augment_image())

val_dataset = ImageDataset(root_dir = image_dir,
                           num_augments = num_augments,
                           phase = 'val',
                           transform = augment_image())

test_dataset = ImageDataset(root_dir = image_dir,
                            num_augments = num_augments,
                            phase = 'test',
                            transform = augment_image())

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)
    
#use_gpu = torch.cuda.is_available()
use_gpu = False

def visualize(images, num_images):
    
    plt.figure()
    
    for i in range(num_images):
        plt.subplot(np.sqrt(num_images),np.sqrt(num_images),i+1)
        plt.imshow(images[i])

    
def train(epochs, encoder, optimizer, criterion):
    
    train_losses = []
    val_losses = []
    best_loss = None
    
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        
        for phase in ['train','val']:
            
            if phase == 'train':
                encoder.train()
                t0 = time.time()
                dataset = train_dataloader
            elif phase == 'val':
                encoder.eval()
                dataset = val_dataloader
            
            for i, images in enumerate(dataset):
                
                current_batch = images.shape[0]
                if use_gpu:
                    encoder = encoder.cuda()
                    images = Variable(images.view(current_batch*num_augments,3,224,224)).cuda()
                else:
                    images = Variable(images.view(current_batch*num_augments,3,224,224))
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    encoder_output = encoder(images).view(current_batch,num_augments,2048)
                    loss = criterion(encoder_output, num_augments, trade_off = 0.25, temperature = 0.05)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()/current_batch
                        
                    elif phase == 'val':
                        val_loss += loss.item()/current_batch
                  
            if phase == 'train':
                t1 = time.time()
                print('Training epoch {} completed in {} with a loss of {}'.format(epoch,t1-t0,train_loss))
                
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if best_loss is None or val_loss < best_loss:
            
            torch.save(encoder, 'best_model')
            
    return train_losses, val_losses

def test():
    
    encoder = torch.load('best_model')
    encoder.eval()
    test_loss = 0
    
    for i,images in enumerate(test_dataloader):
        
        current_batch = images.shape[0]
        if use_gpu:
            encoder = encoder.cuda()
            images = Variable(images.view(current_batch*num_augments,3,224,224)).cuda()
        else:
            images = Variable(images.view(current_batch*num_augments,3,224,224))
            
        with torch.no_grad():
            encoder_output = encoder(images).view(current_batch,num_augments,2048)
            loss = criterion(encoder_output, num_augments, trade_off = 0.25, temperature = 0.05)
            
            test_loss += loss.item()
            
    return test_loss

if __name__ == "__main__":
    
    encoder = construct_encoder()
    optimizer = get_optimizer(encoder, 0.05)
    criterion = loss_fn()
    
    train_losses, val_losses = train(10, encoder, optimizer, criterion)
#    test_loss = 