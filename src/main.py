from .utils.reviewloader import ReviewLoader
from .preprocessing.dataset import prepare_dataset
from .client.client import generate_client_fn

import hydra
from omegaconf import DictConfig, OmegaConf
import flwr
import os

@hydra.main(config_path="../conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    data = None

    # Prepare our dataset
    trainloaders, validationloaders, testloader = prepare_dataset(data, cfg.num_clients, cfg.batch_size)

    # Define clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)




if __name__=="__main__":
    main()


    
