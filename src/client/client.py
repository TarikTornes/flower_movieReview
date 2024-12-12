from flwr.common import NDArrays, Scalar
import torch
from collections import OrderedDict
from ..model.model import Net, train, test
from typing import Dict, Tuple

import flwr as fl


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, trainloader, valloader, num_classes):
        self.trainloader = trainloader
        self.valloader = valloader

        self.model = Net(num_classes)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.device = torch.device(device)


    def set_parameters(self, parameters):
        ''' This function resets the weights/parameters of the model
            to the new parameters given through the input, which represent
            the modified parameters given by the server
        '''

        # FOLLOWING IS A PYTORCH IMPLEMENTATION
        # Thuc needs to adjust to his sentiment analysis model

        params_dict = zip(self.model.state_dict().keys(), parameters)

        # transforms every parameter in the numpy array into a torch tensor representation
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)


    def get_parameters(self, config: Dict[str, Scalar]):
        ''' The following function performs the opposite to set_parameters.
            It takes the model weights and returns them in form of a numpy
            array.

            Args:
                config:

            Return:
                
        '''

        # Following is assumed that the weights of the model are in pytorch tensors
        # Thuc needs to adapt the following depending on the Net model

        # return [val.cpu().model().numpy() for _, val in self.model.state_dict().items()]
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]




    def fit(self, parameters, config):
        ''' Fits/Trains the model in order to retrieve the updated weights

            Args:
                parameters:

            Return:
                return1: new trained weights/parameters
                return2: amount of training sample (can be important 
                         for the aggregation method used in the server side)
                return3: Additional meta-information about the training process
        '''

        # copy parameters sent by the server into clients local model
        self.set_parameters(parameters)

        # Extract from the config all the necessary hyperparameters for the model
        # in order to train

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # perform loacl training
        train(self.model, self.trainloader, optim, epochs, self.device)


        return self.get_parameters(), len(self.trainloader), {}



    def evaluate(self, parameters: NDArrays, config:Dict[str, Scalar]):
        ''' Evaluates how the global models performs on the validation set
            of the client
        '''

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {'accuracy': accuracy}




def generate_client_fn(trainloaders, valloaders, num_classes):

    def client_fn(client_id: str):
        return FlowerClient(trainloader= trainloaders[int(client_id)],
                            valloader=valloaders[int(client_id)],
                            num_classes=num_classes)


    return client_fn
