import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, namedtuple
import pandas as pd

Instance = namedtuple('Instance', ['x', 'y'])

a = Instance('this is a good sentence'.split(), 'positive')


def token_generator(sentences):
    for sentence in sentences:
        for token in sentence:
            yield token


class NLPDataset(Dataset):

    def __init__(self, instances, text_vocab=None, labels_vocab=None, **hyperparams):
        max_size = hyperparams.get('max_size', -1)
        min_freq = hyperparams.get('min_freq', 1)

        if text_vocab is None:
            frequencies = Counter(token_generator(i[0] for i in instances))
            self.text_vocab = Vocab(frequencies, max_size, min_freq)
        else:
            self.text_vocab = text_vocab
        if labels_vocab is None:
            self.labels_vocab = Vocab(Counter(i[1] for i in instances), pad_and_unk=False)
        else:
            self.labels_vocab = labels_vocab

        self.instances = instances

    def __getitem__(self, index) -> Instance:
        instance = self.instances[index]
        return self.text_vocab.encode(instance[0]), self.labels_vocab.encode(instance[1])

    def __len__(self):
        return len(self.instances)

    @classmethod
    def from_file(cls, file_path, text_vocab=None, labels_vocab=None):
        df = pd.read_csv(file_path, header=0, names=['sent', 'label'])
        return cls(
            [Instance(row['sent'].split(), row['label'].strip()) for i, row in df.iterrows()],
            text_vocab=text_vocab,
            labels_vocab=labels_vocab
        )


class Vocab:

    PAD = '<PAD>'
    UNK = '<UNK>'
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, frequencies, max_size=-1, min_freq=1, pad_and_unk=True):
        if pad_and_unk is True:
            self.stoi = {Vocab.PAD: Vocab.PAD_IDX, Vocab.UNK: Vocab.UNK_IDX}
            self.itos = {Vocab.PAD_IDX: Vocab.PAD, Vocab.UNK_IDX: Vocab.UNK}
        else:
            self.stoi = dict()
            self.itos = dict()

        for i, entry in enumerate(sorted(frequencies.items(), key=lambda x: -x[1])):
            token, freq = entry
            # if max_size is set and there is already max_size elements, or if freq is lower than min_freq
            if (max_size != -1 and i >= max_size) or freq < min_freq:
                break

            index = i+2 if pad_and_unk is True else i
            self.stoi[token] = index
            self.itos[index] = token

    def encode(self, tokens):
        if isinstance(tokens, str):
            return torch.tensor(self.stoi.get(tokens, Vocab.UNK_IDX))

        return torch.tensor([self.stoi.get(token, Vocab.UNK_IDX) for token in tokens])

    def decode(self, indices):
        try:
            return [self.itos.get(int(idx), Vocab.UNK) for idx in indices]
        except TypeError:
            return self.itos.get(int(indices), Vocab.UNK)

    def create_embedding_matrix(self, embedding_size=300, path_to_embeddings=None):
        embeddings = torch.randn(len(self.stoi), embedding_size)

        if path_to_embeddings is not None:
            embedding_vocab = dict()
            with open(path_to_embeddings, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.split()
                    embedding_vocab[line[0]] = line[1:]

            for token, index in self.stoi.items():
                embedding = embedding_vocab.get(token, None)
                if embedding is not None:
                    embedding = torch.tensor([float(e) for e in embedding], dtype=torch.float32)
                    embeddings[index] = embedding
        # embedding of <PAD> must be zeros
        embeddings[Vocab.PAD_IDX, :] = 0.0
        return torch.nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)


def pad_collate_fn(batch, pad_index=0, to_cuda=torch.cuda.is_available()):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.int32)
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.tensor(labels, dtype=torch.float32)
    if to_cuda is True:
        lengths = lengths.to('cuda')
        padded_texts = padded_texts.to('cuda')
        labels = labels.to('cuda')
    return padded_texts, labels, lengths


if __name__ == '__main__':
    train_dataset = NLPDataset.from_file('data/sst_train_raw.csv')
    # instance_text, instance_label = train_dataset.instances[2]
    # print(f"Text: {instance_text}")
    # print(f"Label: {instance_label}")
    #
    # numericalized_text, numericalized_label = train_dataset[2]
    # print(f"Numericalized text: {numericalized_text}")
    # print(f"Numericalized label: {numericalized_label}")

    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False, collate_fn=pad_collate_fn)
    # texts, labels, lengths = next(iter(train_dataloader))
    # print(f"Texts: {texts}")
    # print(f"Labels: {labels}")
    # print(f"Lengths: {lengths}")
    #
    # text = train_dataset.instances[1][0]
    # decoded_text = train_dataset.vocab.decode(texts[1])
    # print()
    # print(text)
    # print(decoded_text)
    text_vocab = train_dataset.text_vocab
    embedding = text_vocab.create_embedding_matrix(path_to_embeddings='data/sst_glove_6b_300d.txt')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, collate_fn=pad_collate_fn)

    for batch in train_dataloader:
        batch_x, batch_y, batch_lens = batch
        print(batch_x.size())
        batch_x_embedded = embedding(batch_x)
        print(batch_x_embedded.size())
        batch_x_embedded_averaged = batch_x_embedded.mean(dim=1)
        print(batch_x_embedded_averaged.size())
        break
