import torch
from net import Net
from train import Train
import time
def main():
    print('CUDA: ' + str(torch.cuda.is_available()))
    trening = Train(tl='N',n_epochs=3,imba='Y', lr=0.0001, activate='leakyrelu')
    model = trening.train()
    # TLtrain = Train_TL()
    # model = TLtrain.train()
    torch.save(model.state_dict(), 'CNN.pth')

if __name__ == '__main__':
    start_time = time.time()
    print("-----------START-----------")
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Program execution time: %d seconds" % execution_time)
