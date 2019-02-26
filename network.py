from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
from torch import Tensor
import torch
import torch.nn as nn
import random
import numpy as np
import torchvision.models as models

class BERT(nn.Module):
    def __init__(self, num_labels=58):
        super().__init__()
        
        self.num_labels= num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.resnet = torch.nn.Sequential(*(list(models.resnet152(pretrained=True).children())[:-1]))
        self.resnet_downsample = torch.nn.Linear(2048, 768)
        self.classifier = torch.nn.Linear(768*2, num_labels)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, image=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        if image is not None:
            img = self.resnet(image)
            img = img.view(img.size(0), -1)
            img = self.resnet_downsample(img)
            assert pooled_output.shape[0] == img.shape[0]
            assert len(pooled_output.shape) == len(img.shape)
            pooled_output = torch.cat((pooled_output, img), 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.long())
            return loss
        else:
            return logits