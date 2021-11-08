from itertools import product
import data
import models
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import BCEWithLogitsLoss
from torch import cuda

EMBEDDING_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 5
RESULTS_PATH = 'results/hyperparameters_results.csv'

train_dataset = data.NLPDataset.from_file('data/sst_train_raw.csv')
text_vocab = train_dataset.text_vocab
labels_vocab = train_dataset.labels_vocab
test_dataset = data.NLPDataset.from_file('data/sst_test_raw.csv', text_vocab, labels_vocab)
val_dataset = data.NLPDataset.from_file('data/sst_valid_raw.csv', text_vocab, labels_vocab)

embedding = text_vocab.create_embedding_matrix(EMBEDDING_SIZE, path_to_embeddings='data/sst_glove_6b_300d.txt')

hidden_sizes = [100, 150, 200]
nums_layers = [2, 4, 6]
dropouts = [0, 0.2, 0.5]
bidirectionals = [True, False]

with open(RESULTS_PATH, 'w', encoding='utf-8') as file:
    # header
    print('hidden_size, num_layers, dropout, bidirectional, loss, accuracy, precision, recall, f1', file=file)
    for hidden_size, num_layers, dropout, bidirectional in product(hidden_sizes, nums_layers, dropouts, bidirectionals):

        model = models.RNNModel(
            'lstm',
            embedding,
            input_size=EMBEDDING_SIZE,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        optimizer = Adam(model.parameters(), lr=1e-4)
        scheduler = ExponentialLR(optimizer, gamma=1)
        criterion = BCEWithLogitsLoss()

        if cuda.is_available():
            model.to('cuda')

        for epoch in range(EPOCHS):
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                          collate_fn=data.pad_collate_fn)
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                        collate_fn=data.pad_collate_fn)

            models.train(model, train_dataloader, optimizer, criterion, scheduler, 0.25)
            metrics = models.evaluate(model, val_dataloader, criterion)
            print(f"Epoch {epoch + 1}: validation loss = {metrics['loss']} validation accuracy: {metrics['accuracy']}")

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data.pad_collate_fn)
        metrics = models.evaluate(model, test_dataloader, criterion)
        acc, prc, rec, f1 = metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']
        line = f"{hidden_size}, {num_layers}, {dropout}, {bidirectional}, {metrics['loss']}, {acc}, {prc}, {rec}, {f1}"
        print(line)
        print(line, file=file)
        cuda.empty_cache()
