from torch.utils.data import random_split, DataLoader
import torch

def get_dataset(data):
    '''This function will give us the dataset in a form that
        can be passed through the model.

        Args:
            data: is the dataframe with "text" and "label" column

        Return:
            trainset and testset
    '''

    pass





def prepare_dataset(data, num_partitions: int, 
                    batch_size: int, val_ratio:
                    float = 0.1):
    '''This function creates the a partitions of the dataset such that we
        can simulate the different clients in the FL model.

        Args:
            data: represents the whole dataset in form of a dataframe
            num_partitions: number of partions we want to have (i.e. num of clients)
            batch_size: represents the batch size used by the dataloaders
            val_ration: is the ratio which will be taken from the
                        trainset for the validation set

        Return:
            trainloaders: list of dataloaders, for each client/partition one
            valloaders: list of validation set dataloaders, for each client one
            valloaders: testset dataloader
    '''

    trainset, testset = get_dataset(data)

    # split trainset into 'num_partitions' trainsets
    num_samples = len(trainset) // num_partitions

    partitions_length = [num_samples] * num_partitions

    trainsets = random_split(trainset, partitions_length, torch.Generator().manual_seed(2023))

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], \
                                          torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size,\
                                       shuffle=True, num_workers=2))

        valloaders.append(DataLoader(for_val, batch_size=batch_size,\
                                       shuffle=True, num_workers=2))


    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader








