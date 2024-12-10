from .utils.reviewloader import ReviewLoader
from .preprocessing.dataset import prepare_dataset
from .client.client import generate_client_fn
from .server.server import get_on_fit_config, get_eval_fn

import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl
import os

@hydra.main(config_path="../conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    data = None

    # Prepare our dataset
    trainloaders, validationloaders, testloader = prepare_dataset(data, cfg.num_clients, cfg.batch_size)

    # Define clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)


    # Define the strategy to aggregate the weights
    # fraction_fit will sample a certain fraction of clients for a round
    # by default it is 1.0, we set it very small in order to use min_fit_clients parameter
    # since min_fit_clients will be taken if it is larger than fraction_fit * num_clients
    # therefore we set it very small
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.000000001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.0000001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_eval_fn(cfg.num_classes, testloader))

    # Start Simulation

    history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={'num_cpus': 2, 'num_gpus': 0.5},     # num_gpu -> fraction of gpu
    )




if __name__=="__main__":
    main()


    
