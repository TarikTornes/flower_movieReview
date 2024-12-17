import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_dim  # Fix: Assign to self.hidden_size
        self.num_layers = num_layers  # Fix: Assign to self.num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Apply dropout before fully connected layer
        out = self.fc(out)  # Output from the last time step
        return out


def train(net, trainloader, optimizer, epochs, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)

    for epoch in range(epochs):
        total_train_loss = 0  # Fix: Reset loss at the beginning of each epoch

        for batch in trainloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = net(input_ids)  # Fix: Ensure the input matches the forward method
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_train_loss:.4f}")


def test(net, testloader, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    correct_predictions, total_test_loss, total_predictions = 0, 0.0, 0
    net.eval()
    net.to(device)

    with torch.no_grad():
        for batch in testloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = net(input_ids)  # Fix: Ensure the input matches the forward method
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions  # Fix: Use total_predictions
    return total_test_loss, accuracy
