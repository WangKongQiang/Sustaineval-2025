import time
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import build_iterator, get_time_dif, build_dataset


def submit(config, model):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    data = []
    PAD, CLS = '[PAD]', '[CLS]'
    pad_size = 128
    test_iter = []

    test = pd.read_csv('./data/Evaluation.csv')
    test_text = test['target'].values.tolist()
    test_id = test['id'].values.tolist()

    for line in tqdm(range(0, len(test_text))):
        content = test_text[line]
        token = config.tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        test_iter.append((token_ids, int(1), seq_len, mask))
    test_iter = build_iterator(test_iter, config)

    with torch.no_grad():
        predict_all = np.array([], dtype=int)
        for texts, labels in test_iter:
            # print("texts:",texts)
            # print("labels:", labels)
            outputs= model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
        dir = {
            0: "1",
            1: "2",
            2: "3",
            3: "4",
            4: "5",
            5: "6",
            6: "7",
            7: "8",
            8: "9",
            9: "10",
            10: "11",
            11: "12",
            12: "13",
            13: "14",
            14: "15",
            15: "16",
            16: "17",
            17: "18",
            18: "19",
            19: "20",
        }
        predict_all = [dir[n] for n in predict_all]
        data = {
            'id': test_id,
            'label': predict_all
        }
        print("len(data['id']):",len(data['id']),"len(data['label']):",len(data['label']))
        df = pd.DataFrame(data)
        df.to_csv(r"./data/prediction_task_a.csv", index=False)

if __name__ == '__main__':
    dataset = 'data'
    model_name = 'dbmdz-bert'
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    model = x.Model(config).to(config.device)
    submit(config, model)