import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self): #input 36x36
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 6 * 6, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNN_NET(nn.Module):
    def __init__(self):
        #https://madebyollin.github.io/convnet-calculator/ - calculor for AI
        super(CNN_NET, self).__init__()
        # 1 x 36 x 36
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,8,5,padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # 8 x 18 x 18
        self.conv2 = nn.Sequential(
            nn.Conv2d(8,16,3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # 16 x 9 x 9
        self.conv3 = nn.Sequential(
            nn.Conv2d(16,32,3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3,3)
        )
        # 32 x 3 x 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        # 64 x 3 x 3 = 576
        self.classfier = nn.Sequential(
            nn.Linear(576,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2)
            # nn.Softmax()
        )


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classfier(x)
        return x

