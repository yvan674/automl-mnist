"""Search Worker.

Creates a search worker that hpbandster and use for the MNIST classification
network.
"""
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision
import ConfigSpace as CS
from hpbandster.core.worker import Worker
from os import getcwd
from os.path import join

from network import Network


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class SearchWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get the datasets first
        self.train_data = torchvision.datasets.MNIST(
            root=join(getcwd(), 'mnist'),
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        self.test_data = torchvision.datasets.MNIST(
            root=join(getcwd(), 'mnist'),
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    def compute(self, config: dict, budget: int, **kwargs):
        """Runs the training session.

        Args:
            config: Dictionary containing the configuration
            budget (int): Amount of epochs the model can train for

        Returns:
            dict: dictionary with keys 'loss' (float) and 'info' (dict)
        """
        # Set up configuration from the config dict
        network, criterion, optim, train_loader, test_loader = self.prep(config)

        loss_value = 0
        for epoch in range(int(budget)):
            network.train()
            for data in train_loader:
                optim.zero_grad()
                out = network(data[0].to(DEVICE, non_blocking=True))

                loss = criterion(out, data[1].to(DEVICE, non_blocking=True))

                loss.backward()
                optim.step()
                loss_value = loss.item()

        validation_accuracy, validation_loss = self.evaluate_network(
            network,
            criterion,
            test_loader
        )

        return {'loss': 1 - validation_accuracy,
                'info': {'Final loss': loss_value,
                         'Validation loss': validation_loss.item()}
                }

    def prep(self, config: dict):
        """Runs configuration prep in a separate function for clarity.

        Args:
            config: The configuration dictionary.

        Returns:
            nn.Module: The ann that will be used.
            nn.Module: The loss function.
            nn.Module: The optimizer
            DataLoader: The training data loader
            DataLoader: The testing data loader

        """
        # Configure network
        network = Network(config['num_conv_layers'], config['kernel_size'],
                          config['activation'], config['fc_units'],
                          config['per_layer_dropout'], config['fc_dropout'])

        network = network.to(device=DEVICE)

        # Configure loss
        criterion = CrossEntropyLoss()

        # Configure optimizers
        if config['optimizer'] == 'adagrad':
            optim = torch.optim.Adagrad(network.parameters(),
                                        lr=config['lr'])
        elif config['optimizer'] == 'adam':
            optim = torch.optim.Adam(network.parameters(),
                                     lr=config['lr'],
                                     eps=config['eps'])
        elif config['optimizer'] == 'rmsprop':
            optim = torch.optim.RMSprop(network.parameters(),
                                        lr=config['lr'],
                                        momentum=config['momentum'])
        elif config['optimizer'] == 'sgd':
            optim = torch.optim.SGD(network.parameters(),
                                    lr=config['lr'],
                                    momentum=config['momentum'])
        else:
            raise ValueError("Chosen optimizer is not one of the possible "
                             "options")

        # Configure data loaders
        train_loader = DataLoader(self.train_data,
                                  batch_size=config['batch_size'],
                                  shuffle=True)
        test_loader = DataLoader(self.test_data,
                                 batch_size=config['batch_size'],
                                 shuffle=True)

        return network, criterion, optim, train_loader, test_loader

    def evaluate_network(self, network, criterion, data_loader):
        """Evaluate network accuracy on a specific data set.

        Returns:
            float: Average accuracy
            float: Average loss
        """
        network.eval()
        accuracy = 0.
        loss = 0.
        total_values = float(len(data_loader))

        # Use network, but without updating anything
        with torch.no_grad():
            for data in data_loader:
                out = network(data[0].to(DEVICE, non_blocking=True))

                loss += criterion(out,
                                  data[1].to(DEVICE, non_blocking=True))

                # TODO Calculate batch accuracy


if __name__ == '__main__':
    a = SearchWorker(run_id='0')
    config = {'num_conv_layers': 2,
              'kernel_size': 5,
              'activation': 'relu',
              'per_layer_dropout': 0.5,
              'fc_units': 50,
              'fc_dropout': 0.5,
              'batch_size': 1,
              'lr': 0.01,
              'optimizer': 'sgd',
              'eps': 0.1,
              'momentum': 0.5
    }
    a.compute(config, 1)
