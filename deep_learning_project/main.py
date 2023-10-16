import torch
from net import Net
from load_data import train_loader, valid_loader, test_loader
from train import Train

def main():
    print('CUDA: ' + str(torch.cuda.is_available()))
    trening = Train()
    model = trening.train()

if __name__ == '__main__':
    print("-----------START-----------")
    main()
