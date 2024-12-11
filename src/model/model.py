import torch
import torch.nn as nn
from transformers import BertModel


class Net(nn.Module):
    ''' This represents the model we want to train, and which will be used
        for our federated learning

    '''

    def __init__(self, num_classes: int):
        super(Net, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)



    def forward(self, input_ids, attention_mask, token_type_ids):
        ''' Forward method for the model

            Args:
                x: represents the training data(features)
        '''
        # gets the embeddings from the pretrained bert model
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
        )
        
        # pooler output, represents the content of the 
        # whole sequence (BERT specific [CLS] token, which can be
        # seen as a summary)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)

        return logits


def train(net, trainloader, optimizer, epochs, device: str):

    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)


    for _ in range(epochs):
        total_train_loss = 0
        counter = 0

        for batch in trainloader:
            counter += 1 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = net(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()
            total_train_loss += loss.item()




def test(net, testloader, device:str):
    criterion = torch.nn.CrossEntropyLoss()
    correct_predictions, total_test_loss, total_predictions = 0, 0.0, 0
    net.eval()
    net.to(device)

    with torch.no_grad():
        for batch in testloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = net(input_ids, attention_mask, token_type_ids)


            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / len(testloader.dataset)
    # accuracy = correct_predictions / total_predictions

    return total_test_loss, accuracy

  
