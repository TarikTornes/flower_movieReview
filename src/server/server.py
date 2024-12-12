from omegaconf import DictConfig
from ..model.model import Net, test

from collections import OrderedDict

import torch

def get_on_fit_config(config: DictConfig):
    ''' This function returns a function allows us to configure/adapt 
        certain hyperparameters dependign on the round we are for 
        example increase or decrease the learning rate, etc.

        Args:
            server_round: the round the client is training in

        Return:
            fit_config_fn: Function that contains the constraints depending on the round

    '''

    def fit_config_fn(server_round: int):
        
        # DEFINE SOME CONDITIONS IF NEEDED

        return {'lr': config.lr, 'momentum': config.momentum, 
                'local_epochs': config.local_epochs}

    return fit_config_fn


def get_eval_fn(num_classes, testloader):

    def eval_fn(server_round: int, parameters, config):

        model = Net(num_classes)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        device = torch.device(device)


        # NEEDS TO BE MODIFIED DEPENDING ON MODEL
        # Supposed to set the new parameters for the central model
        # for evaluation
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, acc = test(model, testloader, device)

        return loss, {'loss': loss, 'accuracy': acc}

        

    return eval_fn

