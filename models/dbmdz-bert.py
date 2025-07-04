import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class Config(object):

    def __init__(self, dataset):
        self.model_name = 'dbmdz-bert'
        self.train_path = dataset + '/train.txt'
        self.dev_path = dataset + '/dev.txt'
        self.test_path = dataset + '/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 30
        self.batch_size = 8
        self.pad_size = 128
        self.learning_rate = 2e-5
        self.bert_path = './models--dbmdz--bert-base-german-cased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout=0.2


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)    
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        # output = self.roberta(context, attention_mask=mask)
        output = self.bert(context, attention_mask=mask,output_hidden_states=True)
        # pooled = output[2][10:13][:, 0, :]
        # pooled = output[2][9:13]
        # pooled_mean=(pooled[0][:, 0, :]+pooled[1][:, 0, :]+pooled[2][:, 0, :]+pooled[3][:, 0, :])/4
        # pooled_mean=self.dropout_bertout(pooled_mean)
        # pooled = output[1]
        # out = self.fc(pooled_mean)
        hidden_states=output[2]
        nopooled_output = torch.cat((hidden_states[9], hidden_states[10], hidden_states[11], hidden_states[12]), 1)
        batch_size = nopooled_output.shape[0]
        kernel_hight = nopooled_output.shape[1]
        pooled_output = F.max_pool2d(nopooled_output, kernel_size=(kernel_hight, 1))
        flatten = pooled_output.view(batch_size, -1)
        flattened_output = self.dropout(flatten)
        out = self.fc(flattened_output)
        return out
