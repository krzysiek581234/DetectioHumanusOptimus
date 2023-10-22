import torch
from net import Net
from train import Train
import time
def main():
    print('CUDA: ' + str(torch.cuda.is_available()))
    trening = Train()
    model = trening.train()
    torch.save(model.state_dict(), 'CNN.pth')

if __name__ == '__main__':
    start_time = time.time()
    print("-----------START-----------")
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Program execution time: %d seconds" % execution_time)
