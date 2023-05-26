
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from dataset_utility import dataset, ToTensor

import torch.nn.functional as F
from config import DIR_PATH

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

tf = transforms.Compose([ToTensor()])   

current_path = os.getcwd()
directory_path = os.path.dirname(current_path)

dir_path =  directory_path + DIR_PATH;

param_mode = "r";
param_img_size = 128;
param_batch_size = 32;


train_set = dataset(root = dir_path, dataset_type =  'train', fig_type = "*", img_size = param_img_size, transform =  tf, train_mode = True)
val_set = dataset(root = dir_path, dataset_type =  'val', fig_type = "*", img_size = param_img_size, transform =  tf, train_mode = True)

#print('train length', len(train_set), "*");

if __name__ == '__main__':

    # Create a DataLoader for the dataset
    train_loader = DataLoader(train_set, batch_size=param_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=param_batch_size, shuffle=True, num_workers=4)
    #print(train_loader)

    #Define ResNet18 architecture
    model = models.resnet18(pretrained=True)

    # Freeze the parameters in ResNet18, except for the last fully connected layer:
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512,8) # output layer with 10 classes, set this to the number of classes in your dataset

    # Define the loss function and optimizer:

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    net = Net()
            
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # Set the number of epochs to train for

    for epoch in range(num_epochs):
        print("*" * 30)
        print("Epoch", epoch, "started:")
        # Train the model on the training dataset
        col = 0
        for batch_idx, (resize_image, target) in enumerate(train_loader):
            #print(resize_image.shape)
            col = col + 1
            print("Batch idx : ", col)
            optimizer.zero_grad()  # Clear the gradients
            output = model(resize_image)  # Feed forward
            loss = criterion(output, target)  # Calculate the loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the weights

        # Evaluate the model on the validation dataset
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for resize_image, target in val_loader:
                output = model(resize_image)
                _, predicted = torch.max(output, dim=1)
                num_correct += (predicted == target).sum().item()
                num_samples += predicted.size(0)
            accuracy = num_correct / num_samples
            print(f"Epoch {epoch+1}/{num_epochs}: validation accuracy = {accuracy:.4f}")