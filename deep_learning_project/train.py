import torch
from net import Net, CNN_NET
#from load_data import train_loader, valid_loader, test_loader
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from load_data import data_load
from torchvision import models
# TO DO RESNET18
# hyper do maina
# Relu Leaky Relu
# Imbalanced Data set
class Train:
    def __init__(self, LR = 0.001,patience=5, TL="N") -> None:

        #--------- HYPER PARAMATERS---------
        print("Train Init")
        self.n_epochs = 20
        self.learning_rate = LR
        #train_loader
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs_without_improvement = 0
        self.best_validation_metric = 0
        self.TL = TL

        self.load_data()


    def load_data(self):

        if self.TL == 'N':
            #----------------ATM TRANFORM-------------
            # https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
            transform = torchvision.transforms.Compose(
                [
                torchvision.transforms.RandomPosterize(4, 0.1), #
                torchvision.transforms.RandomHorizontalFlip(p=0.2), # 20% - chance to do that
                torchvision.transforms.RandomVerticalFlip(p=0.2), #
                torchvision.transforms.RandomEqualize(p=0.1), #
                torchvision.transforms.RandomInvert(p=0.1),
                
                torchvision.transforms.RandomApply(nn.ModuleList([torchvision.transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), shear=10),]), p=0.2),
                torchvision.transforms.RandomApply(nn.ModuleList([torchvision.transforms.GaussianBlur(kernel_size=5),]), p=0.24),
                torchvision.transforms.RandomApply(nn.ModuleList([torchvision.transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1, 1.5), saturation=(0.5, 1.5),hue=(-0.1, 0.1)),]), p=0.2),


                torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur((1, 9), (0.2,0.3))],p=0.2),
                transforms.Grayscale(),
                transforms.ToTensor(),   
                transforms.Normalize(mean=(0.5,),std=(0.5,))]) #
            self.data = data_load(transform)
        else:
            self.transform = torchvision.transforms.Compose(
            [
            torchvision.transforms.RandomPosterize(4, 0.1), #
            torchvision.transforms.RandomHorizontalFlip(p=0.2), # 20% - chance to do that
            torchvision.transforms.RandomVerticalFlip(p=0.2), #
            torchvision.transforms.RandomEqualize(p=0.1), #
            torchvision.transforms.RandomInvert(p=0.1),
            
            torchvision.transforms.RandomApply(nn.ModuleList([torchvision.transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), shear=10),]), p=0.2),
            torchvision.transforms.RandomApply(nn.ModuleList([torchvision.transforms.GaussianBlur(kernel_size=5),]), p=0.24),
            torchvision.transforms.RandomApply(nn.ModuleList([torchvision.transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1, 1.5), saturation=(0.5, 1.5),hue=(-0.1, 0.1)),]), p=0.2),


            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur((1, 9), (0.2,0.3))],p=0.2),
            transforms.ToTensor(),   
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.data = data_load(self.transform, TL='Y')

        

    def train(self, optim='Adam', WD = 0.0,log='N'):

        print("Training")
        if self.TL == 'Y':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, 2)
        else:
            model = CNN_NET()
        model.to(self.device)

        if(optim == 'Adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=WD)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=WD)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(device=self.device),  reduction='mean')
        n_correct = 0
        n_samples = 0


        for epoch in range(0, self.n_epochs):
            #train_loader
            for i, data in enumerate(self.data.train_loader):
                images, labels = data
                optimizer.zero_grad()
                images = images.to(self.device)
                # forward
                outputs = model(images)
                labels = labels.to(self.device)
                loss = criterion(outputs, labels.type(torch.int64))

                #backwards
                loss.backward()
                optimizer.step()

                #calculate traning accuracy
                _, predictions = torch.max(outputs, 1)
                n_samples += labels.shape[0]
                n_correct += (predictions == labels).sum().item()

                if(i+1) % 100 == 0:
                    print(f"epoch {epoch +1} / { self.n_epochs}, step {i+1}/{self.data.n_total}, loss: {loss}")

            if self.best_validation_metric < self.validate(model):
                self.epochs_without_improvement =0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement > 5:
                    return model
        return model
    
    def validate(self, model):
        with torch.no_grad():
            print('Validation: ')
            correct = 0
            total = 0
            total_face = 0
            total_wall = 0
            correct_face = 0
            correct_wall = 0
            for data in self.data.test_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_face += (labels == 1).sum().item()  # face class is labeled as 1
                total_wall += (labels == 0).sum().item()  # wall class is labeled as 0
                correct_face += ((predicted == 1) & (labels == 1)).sum().item()
                correct_wall += ((predicted == 0) & (labels == 0)).sum().item()
            print("correct face %d %%" % (100 * correct_face / total_face))
            print("correct walls %d %%" % (100 * correct_wall / total_wall))

            # correct face 732 / 797
            # correct walls 4415 / 6831

            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
            return correct
