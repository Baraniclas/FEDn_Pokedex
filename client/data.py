import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(is_train=True):
    """
    Load data from disk.

    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tensor of data and target.
    :rtype: torch.tensor
    """

    " change to /app/data/water.csv "
    data_path = '/home/baraniclas/FEDn_Pokedex/data/water.csv'

    water = pd.read_csv(data_path)

    X = water.drop(columns=['Speed', 'Name', 'Unnamed: 0'])
    y = water['Speed']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    if is_train:
        return X_train_tensor, y_train_tensor
    else:
        return X_test_tensor, y_test_tensor