import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif,build_iterator
from pytorch_pretrained_bert.optimization import BertAdam
import pandas as pd
from tqdm import tqdm
import transformers


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.1,
                         t_total=len(train_iter) * config.num_epochs)

    # num_train_optimization_steps = len(train_iter) * config.num_epochs
    # optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=True)
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
    #                                                          int(num_train_optimization_steps * 0.1),
    #                                                          num_train_optimization_steps)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    model.train()

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs= model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

# def submit(config, model):
#     # test
#     model.load_state_dict(torch.load(config.save_path))
#     model.eval()
#
#     data = []
#     PAD, CLS = '[PAD]', '[CLS]'
#     pad_size = 64
#     test_iter = []
#
#     test = pd.read_csv('./data/dev_task_a_entries.csv')
#     test_text = test['text'].values.tolist()
#     test_id = test['rewire_id'].values.tolist()
#
#     for line in tqdm(range(0, len(test_text))):
#         content = test_text[line]
#         token = config.tokenizer.tokenize(content)
#         token = [CLS] + token
#         seq_len = len(token)
#         mask = []
#         token_ids = config.tokenizer.convert_tokens_to_ids(token)
#
#         if pad_size:
#             if len(token) < pad_size:
#                 mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
#                 token_ids += ([0] * (pad_size - len(token)))
#             else:
#                 mask = [1] * pad_size
#                 token_ids = token_ids[:pad_size]
#                 seq_len = pad_size
#         test_iter.append((token_ids, int(1), seq_len, mask))
#     test_iter = build_iterator(test_iter, config)
#
#     with torch.no_grad():
#         predict_all = np.array([], dtype=int)
#
#         with torch.no_grad():
#             for texts, labels in test_iter:
#                 outputs= model(texts)
#                 predic = torch.max(outputs.data, 1)[1].cpu().numpy()
#                 predict_all = np.append(predict_all, predic)
#
#     dir = {
#         0: "not sexist",
#         1: "sexist"
#     }
#     predict_all = [dir[n] for n in predict_all]
#     data = {
#         'rewire_id': test_id,
#         'label_pred': predict_all
#     }
#     df = pd.DataFrame(data)
#     df.to_csv(r"./data/dev_task_a_submit_"+config.model_name+".csv", index=False)