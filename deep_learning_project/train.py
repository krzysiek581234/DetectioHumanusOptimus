import torch
from net import Net
from load_data import train_loader, valid_loader, test_loader
import torch.nn as nn

class Train:
    def __init__(self, LR = 0.001,patience=5) -> None:
        #--------- HYPER PARAMATERS---------
        print("Train Init")
        self.n_epochs = 10
        self.learning_rate = LR
        self.n_total = len(train_loader)
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs_without_improvement = 0
        self.best_validation_metric = 0

    def train(self, optim='Adam', WD = 0.0):
        print("Training")
        model = Net()
        model.to(self.device)

        if(optim == 'Adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=WD)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=WD)


        
        criterion = nn.CrossEntropyLoss()
        n_correct = 0
        n_samples = 0
        for epoch in range(0, self.n_epochs):
            for i, data in enumerate(train_loader):
                images, labels = data
                optimizer.zero_grad()

                # forward
                outputs = model(images)
                loss = criterion(outputs, labels.type(torch.int64))

                #backwards
                loss.backward()
                optimizer.step()

                #calculate traning accuracy
                _, predictions = torch.max(outputs, 1)
                n_samples += labels.shape[0]
                n_correct += (predictions == labels).sum().item()

                if(i+1) % 100 == 0:
                    print(f"epoch {epoch +1} / { self.n_epochs}, step {i+1}/{self.n_total}, loss: {loss}")

            if self.best_validation_metric < self.validate(model):
                self.epochs_without_improvement =0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement > 5:
                    break
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
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_face += (labels == 0).sum().item()  # Assuming face class is labeled as 0
                total_wall += (labels == 1).sum().item()  # Assuming wall class is labeled as 1
                correct_face += ((predicted == 0) & (labels == 0)).sum().item()
                correct_wall += ((predicted == 1) & (labels == 1)).sum().item()
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
            return correct

# for data, target in train_loader:






