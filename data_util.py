import torchvision.transforms as transforms

def augment_image():
    
    transforms_image = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, hue=0.25, saturation=0.25),
        transforms.RandomHorizontalFlip(p=0.5),
#        transforms.RandomRotation(90),
        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms_image

