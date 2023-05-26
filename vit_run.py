import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from VisionTransformer import VisionTransformer
from dataset_utility import dataset, ToTensor
import os
from config import DIR_PATH
from torch.utils.data import DataLoader

# Set device to GPU if available or CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to the GPU
model = VisionTransformer(in_channels=3,num_classes=10, patch_size=16, embedding_dim=256, num_layers=12, num_heads=8, mlp_ratio=4)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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



    # Train the model
    epochs = 2
    for epoch in range(epochs):
        print("Epoch ", epoch,"Started");
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            print("Batch ", i);
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
            