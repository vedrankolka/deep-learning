import argparse
import data
import models
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import BCEWithLogitsLoss
from torch import cuda

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1-1e-5)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--clip', type=float, default=None)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--rnn', choices=['vanilla', 'lstm', 'gru'], default='lstm')
    parser.add_argument('--bidirectional', action='store_true', default=False)
    args = parser.parse_args()

    train_dataset = data.NLPDataset.from_file('data/sst_train_raw.csv')
    text_vocab = train_dataset.text_vocab
    labels_vocab = train_dataset.labels_vocab
    test_dataset = data.NLPDataset.from_file('data/sst_test_raw.csv', text_vocab, labels_vocab)
    val_dataset = data.NLPDataset.from_file('data/sst_valid_raw.csv', text_vocab, labels_vocab)

    embedding = text_vocab.create_embedding_matrix(args.embedding_size, path_to_embeddings='data/sst_glove_6b_300d.txt')
    # model = models.AvgPoolingModel(embedding, embedding_size=args.embedding_size)
    model = models.RNNModel(
        args.rnn,
        embedding,
        input_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
    )

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    criterion = BCEWithLogitsLoss()

    if cuda.is_available():
        print("Using GPU")
        model.to('cuda')

    for epoch in range(args.epochs):
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=data.pad_collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data.pad_collate_fn)

        models.train(model, train_dataloader, optimizer, criterion, scheduler, args.clip)
        metrics = models.evaluate(model, val_dataloader, criterion)
        print(f"Epoch {epoch+1}: validation loss = {metrics['loss']} validation accuracy: {metrics['accuracy']}")

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data.pad_collate_fn)
    metrics = models.evaluate(model, test_dataloader, criterion)
    print()
    print(f"Test loss = {metrics['loss']} test accuracy = {metrics['accuracy']}")
    cuda.empty_cache()
