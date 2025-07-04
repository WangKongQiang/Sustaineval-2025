import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'


def load_dataset(path, config):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            if label== 'class1':
                labels = 0
            elif label== 'class2':
                labels = 1
            elif label== 'class3':
                labels = 2
            elif label== 'class4':
                labels = 3
            elif label== 'class5':
                labels = 4
            elif label== 'class6':
                labels = 5
            elif label== 'class7':
                labels = 6
            elif label== 'class8':
                labels = 7
            elif label== 'class9':
                labels = 8
            elif label== 'class10':
                labels = 9
            elif label== 'class11':
                labels = 10
            elif label== 'class12':
                labels = 11
            elif label== 'class13':
                labels = 12
            elif label== 'class14':
                labels = 13
            elif label== 'class15':
                labels = 14
            elif label== 'class16':
                labels = 15
            elif label== 'class17':
                labels = 16
            elif label== 'class18':
                labels = 17
            elif label== 'class19':
                labels = 18
            else:
                labels = 19
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            pad_size = config.pad_size

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append([token_ids, labels, seq_len, mask])
    return contents

def build_dataset(config):
    train = load_dataset(config.train_path, config)
    dev = load_dataset(config.dev_path, config)
    test = load_dataset(config.test_path, config)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size
        self.residue = False
        if len(dataset) % self.batch_size != 0:
            self.residue = True
            print("self.residue:",self.residue)
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
        return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches+1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):

    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
