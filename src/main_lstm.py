from .preprocessing.dataset import prepare_dataset
from .model.lstm import *


trainloaders, valloaders, testloader = prepare_dataset()
