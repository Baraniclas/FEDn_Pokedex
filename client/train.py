from data import load_data
import json
import math
import os
import sys

import torch
from model import load_parameters, save_parameters
from fedn.utils.helpers.helpers import save_metadata
from torch.utils.data import TensorDataset, DataLoader

# swap this to the load_metadata from helpers.helpers on release..
def load_client_settings(filename):
    """Load client settings from file.

    :param filename: The name of the file to load from.
    :type filename: str
    :return: The loaded metadata.
    :rtype: dict
    """
    with open(filename + "-metadata", "r") as infile:
        metadata = json.load(infile)
    return metadata

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

def validate(model, data_path):
    """Validate a model."""
    x_test, y_test = load_data(data_path, is_train=False)
    model.eval()

    criterion_mse = torch.nn.MSELoss()
    with torch.no_grad():
        test_out = model(x_test)
        mse_loss = criterion_mse(test_out, y_test)
    return mse_loss

def train(in_model_path, out_model_path, data_path=None, batch_size=10, epochs = 5, lr=0.01):
    """Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """

    x_train, y_train = load_data()

    model = load_parameters(in_model_path)

    data = TensorDataset(x_train, y_train)
    loader = DataLoader(data, batch_size = batch_size, shuffle = True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for e in range(epochs):
        model.train()
    
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(x_train),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }

    save_metadata(metadata, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])