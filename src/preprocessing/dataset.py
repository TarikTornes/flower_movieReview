from ..utils.reviewloader import load_reviews
from .moviereviewdataset import MovieReviewDataset

from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader
import torch, os, pickle

def get_dataset():
    '''This function will give us the dataset in a form that
        can be passed through the model.

        Return:
            trainset: training set implemented with torch Dataset
            testset: test set implemented with torch Dataset
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    label_encoder = LabelEncoder()


    reviews, labels = load_reviews()
    labels = label_encoder.fit_transform(labels)


    X_train, X_test, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=0.2,
        random_state=42
    )

    trainset = MovieReviewDataset(X_train, y_train, tokenizer)
    testset = MovieReviewDataset(X_test, y_test, tokenizer)

    # trainset= DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
    # testset = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return trainset, testset




def prepare_dataset(num_partitions: int, 
                    batch_size: int, val_ratio:
                    float = 0.1):
    '''This function creates the a partitions of the dataset such that we
        can simulate the different clients in the FL model.

        Args:
            num_partitions: number of partions we want to have (i.e. num of clients)
            batch_size: represents the batch size used by the dataloaders
            val_ration: is the ratio which will be taken from the
                        trainset for the validation set

        Return:
            trainloaders: list of dataloaders, for each client/partition one
            valloaders: list of validation set dataloaders, for each client one
            valloaders: testset dataloader
    '''

    trainset, testset = get_dataset()

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








