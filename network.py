from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
from torch import Tensor
import torch
import torch.nn as nn
import random
import numpy as np
import torchvision.models as models

# resnet_dict= {'resnet18':[models.resnet152(pretrained=True), 512], 'resnet34':[models.resnet34(pretrained=True), 512], 'resnet50':[models.resnet50(pretrained=True), 2048], 'resnet101':[models.resnet101(pretrained=True), 2048], 'resnet152':[models.resnet152(pretrained=True), 2048]}

class BERT(nn.Module):
    def __init__(self, opt, num_labels=58):
        super().__init__()
        
        self.use_images = opt.images
        self.no_bert= opt.no_bert
        self.num_labels= num_labels
        if not self.no_bert:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            if opt.freeze_bert:
                freeze_layer(self.bert)
        self.dropout = nn.Dropout(0.1)
        if self.use_images:
            self.resnet = torch.nn.Sequential(*(list(models.resnet152(pretrained=True).children())[:-1]))
            self.resnet_downsample = torch.nn.Linear(2048, opt.last_layer_size)
        if self.no_bert:
            self.classifier = torch.nn.Linear(opt.last_layer_size, num_labels)
        elif self.use_images:
            self.classifier = torch.nn.Linear(768+opt.last_layer_size, num_labels)
        else:
            self.test = torch.nn.Linear(768, num_labels)


    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, image=None):
        if not self.no_bert:
            _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        if self.use_images:
            img = self.resnet(image)
            img = img.view(img.size(0), -1)
            img = self.resnet_downsample(img)
            if not self.no_bert:
                assert pooled_output.shape[0] == img.shape[0]
                assert len(pooled_output.shape) == len(img.shape)
                pooled_output = torch.cat((pooled_output, img), 1)
            else:
                pooled_output = img
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.long())
            return loss
        else:
            return logits
            
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False