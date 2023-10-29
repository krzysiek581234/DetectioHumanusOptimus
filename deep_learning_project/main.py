import torch
from net import Net
from train import Train
import time
import numpy as np
def main():
    print('CUDA: ' + str(torch.cuda.is_available()))

    start = 0.00001
    stop = 0.001
    # num_steps = 3
    # logarithmic_values = np.logspace(np.log10(start), np.log10(stop), num_steps)

    # for i in logarithmic_values:
    #     trening = Train(tl='N',n_epochs=20,imba='Y', lr=i, activate='leakyrelu', optim='Adam')
    #     print("-----------------------------")
    #     trening.list_self_variables()
    #     print("-----------------------------")
    #     model = trening.train()

    # for i in logarithmic_values:
    #     trening = Train(tl='N',n_epochs=20,imba='Y', lr=i, activate='leakyrelu', optim='SDG')
    #     print("-----------------------------")
    #     trening.list_self_variables()
    #     print("-----------------------------")
    #     model = trening.train()

    trening = Train(tl='N',n_epochs=20,imba='Y', lr=0.001, activate='leakyrelu', optim='Adam')
    print("-----------------------------")
    trening.list_self_variables()
    print("-----------------------------")
    model = trening.train()


    #torch.save(model1.state_dict(), 'CNN.pth')
    #tensorboard --logdir runs
if __name__ == '__main__':
    start_time = time.time()
    print("-----------START-----------")
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Program execution time: %d seconds" % execution_time)
