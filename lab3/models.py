import torch
from torch import nn


class AvgPoolingModel(nn.Module):
    
    def __init__(self, embedding, embedding_size=300):
        super(AvgPoolingModel, self).__init__()
        self.embedding = embedding
        self.linear1 = nn.Linear(embedding_size, 150)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(150, 150)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(150, 1)

    def forward(self, batch, lens):
        batch_embedded = self.embedding(batch)
        # sum across all words
        batch_averaged = batch_embedded.sum(dim=1)
        # divide each result by the number of words
        for i in range(len(batch)):
            batch_averaged[i] /= lens[i]
        h1 = self.relu1(self.linear1(batch_averaged))
        h2 = self.relu2(self.linear2(h1))
        h3 = self.linear3(h2)
        return torch.reshape(h3, (h3.shape[0],))


class LSTMModel(nn.Module):
    def __init__(self, embedding, input_size=300, hidden_size=150, num_layers=2):
        super(LSTMModel, self).__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 150)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(150, 1)

    def forward(self, batch, lens):
        # batch.shape = (batch_size, seq_len, emb_size)
        batch = self.embedding(batch)
        # output.shape = (batch_size, seq_len, hidden_size)
        # hn.shape = cn.shape = (num_layers, batch_size, hidden_size)
        output, (hn, cn) = self.lstm(batch)
        # samo zadnji layer (-1), reshape da budu prve dvije dim
        h1 = hn[-1, :, :].reshape((hn.shape[1], hn.shape[2]))
        h2 = self.relu(self.linear1(h1))
        # h3.shape = (batch_size, 1)
        h3 = self.linear2(h2)
        return h3.reshape((h3.shape[0],))

    def forward_time_first(self, batch, lens):
        """This one is time-first and it performs way worse. No idea why. Also batch_first must be False then."""
        # batch.shape = (batch_size, seq_len, emb_size)
        batch = self.embedding(batch)
        batch_size, seq_len, emb_size = batch.shape
        batch = batch.reshape((seq_len, batch_size, emb_size))
        # output.shape = (batch_size, seq_len, hidden_size)
        # hn.shape = cn.shape = (num_layers, batch_size, hidden_size)
        output, (hn, cn) = self.lstm(batch)
        # samo zadnji layer (-1), reshape da budu prve dvije dim
        h1 = hn[-1, :, :].reshape((hn.shape[1], hn.shape[2]))
        h2 = self.relu(self.linear1(h1))
        # h3.shape = (batch_size, 1)
        h3 = self.linear2(h2)
        return h3.reshape((h3.shape[0],))


class RNNModel(nn.Module):

    def vanilla_rnn_forward(self, batch: torch.Tensor) -> torch.Tensor:
        # output.shape = (batch_size, seq_len, hidden_size)
        # hn.shape = (num_layers, batch_size, hidden_size)
        output, hn = self.rnn(batch)
        h1 = hn[-1].reshape((hn.shape[1], hn.shape[2]))
        return h1

    def lstm_forward(self, batch: torch.Tensor) -> torch.Tensor:
        # output.shape = (batch_size, seq_len, hidden_size)
        # hn.shape = cn.shape = (num_layers, batch_size, hidden_size)
        output, (hn, cn) = self.rnn(batch)
        # samo zadnji layer (-1), reshape da budu prve dvije dim
        h1 = hn[-1].reshape((hn.shape[1], hn.shape[2]))
        return h1

    rnn_cells = {'vanilla': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
    rnn_forwards = {'vanilla': vanilla_rnn_forward, 'lstm': lstm_forward, 'gru': vanilla_rnn_forward}

    def __init__(self, cell_name, embedding, input_size=300, hidden_size=150, num_layers=2, bidirectional=False, dropout=0.0):
        super(RNNModel, self).__init__()
        rnn = RNNModel.rnn_cells[cell_name]
        self.embedding = embedding
        self.rnn = rnn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.rnn_forward = RNNModel.rnn_forwards[cell_name]
        self.linear1 = nn.Linear(hidden_size, 150)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(150, 1)

    def forward(self, batch, lens):
        # batch.shape = (batch_size, seq_len, emb_size)
        batch = self.embedding(batch)
        h1 = self.rnn_forward(self, batch)

        h2 = self.relu(self.linear1(h1))
        # h3.shape = (batch_size, 1)
        h3 = self.linear2(h2)
        return h3.reshape((h3.shape[0],))

    # def rnn_forward(self, batch: torch.Tensor) -> torch.Tensor:
    #     """RNN forward pass.
    #
    #     :param torch.Tensor batch: of shape (batch_size, sequence_len, embedding_size)
    #     :return: the last hidden state in shape = (batch_size, hidden_size)
    #     :rtype: torch.Tensor
    #     """
    #     pass


def train(model, data, optimizer, criterion, scheduler=None, clip=None):
    model.train()
    for batch_index, batch in enumerate(data):
        x, y, lens = batch
        model.zero_grad()
        logits = model.forward(x, lens)
        loss = criterion(logits, y)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()


def evaluate(model, data, criterion):
    model.eval()
    sum_val_loss = 0.0
    counter = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, y, lens = batch
            logits = model(x, lens)
            loss = criterion(logits, y)
            sum_val_loss += loss
            counter += 1

            y_pred = (logits >= 0.0).int()
            tp += torch.sum(torch.logical_and(y_pred == 1, y == 1))
            tn += torch.sum(torch.logical_and(y_pred == 0, y == 0))
            fp += torch.sum(torch.logical_and(y_pred == 1, y == 0))
            fn += torch.sum(torch.logical_and(y_pred == 0, y == 1))

    avg_val_loss = sum_val_loss / counter
    accuracy = (tp + tn)/ (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    confusion_matrix = torch.tensor([[tp, fp], [fn, tn]])
    return {
        "loss": avg_val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix
    }



