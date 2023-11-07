import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

#Train + test file paths
TRAIN_DATA_PATH = "./dataset/asl_alphabet_train"
TEST_DATA_PATH = "./dataset/asl_alphabet_test"


#Define data transformations
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

# Load training set
train_data = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=False)
train_data_loader = DataLoader(train_data, batch_size = 64, shuffle = True)

# Load test set
test_data = datasets.ImageFolder(root=TEST_DATA_PATH, transform = False)
test_data_loader = DataLoader(test_data, batch_size = 64, shuffle=False)

#CNN Model
# 2 conv., 1 max pooling (2,2), 2 FC layers
class CNN(nn.Module):
    
    #constructor
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(1,64,3)
        self.pool = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(p=.25)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.drop2 = nn.Dropout(p=.5)
        self.fc2 = nn.Linear(128, 25)
    
    def forward(self, x):
        # x = tensor for image
        # create conv layers, relu on both, then pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop1(self.conv2(x))
        #resize for FC layer
        x = x.view(-1, 32*13*13)
        x = F.relu(self.fc1(x))
        x = self.drop2(self.fc1(x))
        x = F.softmax(self.fc2(x))
        x = self.fc2(x)
        
        return x

model = CNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    running_loss = 0
    for images, labels in train_data_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch: {epoch+1} - Training loss: {running_loss/len(train_data_loader)}")
    
correct_count, all_count = 0, 0
for images, labels in test_data_loader:
    for i in range(len(labels)):
        img = images[i].view(1,1,28,28)
        with torch.no_grad():
            logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if true_label == pred_label:
            correct_count +=1
        all_count += 1
print(f"# of images tested + {all_count}")
print(f"Model Accuracy = {(correct_count/all_count):.2f}")
