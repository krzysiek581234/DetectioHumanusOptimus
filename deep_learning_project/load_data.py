import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

from torchsampler import ImbalancedDatasetSampler

class data_load():
    def __init__(self, transform, TL = 'N', imba = 'N') -> None:
        self.train_dir = './train_images'    # folder containing training images
        self.test_dir = './test_images'    # folder containing test images
        self.transform = transform
        self.TL = TL
        self.IMBA = imba

        if self.TL == 'N':
            self.basetransform = transforms.Compose(
                [transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
                transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
                transforms.Normalize(mean=(0.5,),std=(0.5,))]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)
        else:
            self.basetransform = transforms.Compose(
                [
                transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)

        train_data = torchvision.datasets.ImageFolder(self.train_dir, transform=self.transform)
        test_data = torchvision.datasets.ImageFolder(self.test_dir, transform=self.basetransform)

        valid_size = 0.2   # proportion of validation set (80% train, 20% validation)
        batch_size = 256 

        # Define randomly the indices of examples to use for training and for validation
        num_train = len(train_data)
        indices_train = list(range(num_train))
        np.random.shuffle(indices_train)
        split_tv = int(np.floor(valid_size * num_train))
        train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

        # Define two "samplers" that will randomly pick examples from the training and validation set

        if self.IMBA == 'N':
            train_sampler = SubsetRandomSampler(train_new_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        else:
            train_sampler = ImbalancedDatasetSampler(train_new_idx)
            valid_sampler = ImbalancedDatasetSampler(valid_idx)



        # Dataloaders (take care of loading the data from disk, batch by batch, during training)
        print("Loading Data")
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,sampler=train_sampler, num_workers=8)
        self.valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,sampler=valid_sampler, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        self.n_total = len(self.train_loader)
        #self.display()

    def display(self):
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)  # Get a batch of images and labels

        # Define the class list
        classes = ('noface', 'face')


        image = images[0].numpy()  # Convert the image tensor to a numpy array
        image = (image * 0.5) + 0.5  # Reverse the normalization

        if image.shape[0] == 1:
            image = image.squeeze()

        # Get the corresponding label
        label = labels[0].item()
        class_name = classes[label]

        # Display the transformed image and label
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))             # Transpose the image to (36, 36, 3) format for RGB
            
        plt.imshow(image, cmap='gray')  # Use 'gray' colormap for grayscale images
        plt.title(class_name)
        plt.show()





























# import numpy as np
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data.sampler import SubsetRandomSampler

# train_dir = './train_images'    # folder containing training images
# test_dir = './test_images'    # folder containing test images

# transform = transforms.Compose(
#     [transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
#      transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
#      transforms.Normalize(mean=(0.5,),std=(0.5,))]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)

# # Define two pytorch datasets (train/test) 
# train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
# test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

# valid_size = 0.2   # proportion of validation set (80% train, 20% validation)
# batch_size = 32    

# # Define randomly the indices of examples to use for training and for validation
# num_train = len(train_data)
# indices_train = list(range(num_train))
# np.random.shuffle(indices_train)
# split_tv = int(np.floor(valid_size * num_train))
# train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

# # Define two "samplers" that will randomly pick examples from the training and validation set
# train_sampler = SubsetRandomSampler(train_new_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)

# # Dataloaders (take care of loading the data from disk, batch by batch, during training)
# print("Loading Data")
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)
# valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

# classes = ('noface','face')  # indicates that "1" means "face" and "0" non-face (only used for display)


# class CustomImageDataset:
#     def __init__(self, train_dir, test_dir, batch_size, valid_size=0.2):
#         self.train_dir = train_dir
#         self.test_dir = test_dir
#         self.transform = transforms.Compose([
#             transforms.Grayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0,), std=(1,))
#         ])
#         self.batch_size = batch_size
#         self.valid_size = valid_size
        
#     def _load_data(self):
#         train_data = torchvision.datasets.ImageFolder(self.train_dir, transform=self.transform)
#         test_data = torchvision.datasets.ImageFolder(self.test_dir, transform=self.transform)

#         num_train = len(train_data)
#         indices_train = list(range(num_train))
#         np.random.shuffle(indices_train)
#         split_tv = int(np.floor(self.valid_size * num_train))
#         train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

#         train_sampler = SubsetRandomSampler(train_new_idx)
#         valid_sampler = SubsetRandomSampler(valid_idx)

#         self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler, num_workers=1, pin_memory=True)
#         self.valid_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=valid_sampler, num_workers=1, pin_memory=True)
#         self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True)

#         self.classes = ('noface','face')
    # def __init(self):
    #     self.train_dir = './train_images'    # folder containing training images
    #     self.test_dir = './test_images'    # folder containing test images
    #     self.transform = transforms.Compose([
    #         transforms.Grayscale(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0,), std=(1,))
    #     ])
    #     self.data = []

    # def __getitem__(self, idx):
    # def __len__(self):


# train_dir = './train_images'
# test_dir = './test_images'

# transform = transforms.Compose(
#     [transforms.Grayscale(), 
#      transforms.ToTensor(), 
#      transforms.Normalize(mean=(0,),std=(1,))])

# train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
# test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

# valid_size = 0.2
# batch_size = 32

# num_train = len(train_data)
# indices_train = list(range(num_train))
# np.random.shuffle(indices_train)
# split_tv = int(np.floor(valid_size * num_train))
# train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

# train_sampler = SubsetRandomSampler(train_new_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1, pin_memory=True)
# valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1, pin_memory=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

# classes = ('noface','face')
