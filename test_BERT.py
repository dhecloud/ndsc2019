import datetime
import ast
import logging
from dataset import *
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
from torch import Tensor
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm, trange
import pickle
from dataloaders import *
from network import *
import argparse
import copy

parser = argparse.ArgumentParser(description='ndsc')
parser.add_argument('--filename', type=str, default='train.csv', help='train csv filename')
parser.add_argument('--expand', action='store_true', help='expand dataset?')
parser.add_argument('--epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--batchsize', type=int, default=32, help='train batch size')
parser.add_argument('--est', type=int, default=5, help='early stopping')
parser.add_argument('--downsample', type=int, default=1000, help='number to downsample imbalanced classes')
parser.add_argument('--num_classes', type=int, default=58, help='number of clases')
parser.add_argument('--images', action='store_true', help='dont use images to train')
parser.add_argument('--resnet', type=str, default='resnet152',choices= ['resnet50', 'resnet152'], help='choice of resnet')
parser.add_argument('--no_bert', action='store_true', help='dont use bert')
parser.add_argument('--freeze_bert', action='store_true', help='freeze bert')
parser.add_argument('--multilingual', action='store_true', help='multilingual bert')
parser.add_argument('--weighted', action='store_true', help='weighted')
parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')

parser.add_argument('--save_dir', type=str, default="experiments/", help='path/to/save_dir - default:experiments/')
parser.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models. if none, it will be saved as the date and time')

parser.add_argument('--last_layer_size', type=int, default=768, help='last layer size for resnet')
parser.add_argument('--fp16', action='store_true', help='floating point support?')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='optimizer gradient accumulation')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--warmup_proportion', type=float, default=0.1, help='optimizer warmup')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning rate')
parser.add_argument('--do_lower_case', type=bool, default=True, help='tokenizer lower case')
parser.add_argument('--max_seq_length', type=int, default=64, help='max sequence length')
    
def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    # save to the disk
    mkdirs(opt.save_path)
    file_name = os.path.join(opt.save_path, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
        
def set_default_opt(opt):
    if not opt.name:
        now = datetime.datetime.now()
        opt.name = now.strftime("%Y-%m-%d-%H-%M")
    opt.save_path = os.path.join(opt.save_dir,opt.name)
    opt.eval_batch_size = opt.batchsize
    opt.train_batch_size = opt.batchsize
    
    return opt
    
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def fit(opt, train_dataloader, optimizer, eval_examples):
    
    global_step = 0
    cur_epoch = 0
    best = 0
    thres = 0
    model.train()
    for i_ in tqdm(range(int(opt.epoch)), desc="Epoch"):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids, img = batch
            img = img.permute(0,3,1,2)
            assert img.shape[1:] == (3,224,224)
            loss = model(input_ids, segment_ids, input_mask, label_ids, image=img)
            if opt.gradient_accumulation_steps > 1:
                loss = loss / opt.gradient_accumulation_steps

            if opt.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % opt.gradient_accumulation_steps == 0:
    #             scheduler.batch_step()
                # modify learning rate with special warm up BERT uses
                lr_this_step = opt.learning_rate * warmup_linear(global_step/t_total, opt.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            state = {
                'epoch': i_,
                'arch': "QAC",
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }


        logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
        logger.info('Eval after epoc {}'.format(i_+1))
        save_checkpoint(state, False, opt, filename=str(cur_epoch)+"_checkpoint.pth.tar")
        cur_epoch += 1 
        eval_acc = eval(eval_examples, opt)
        if eval_acc['top1'] > best:
            save_checkpoint(state, True, opt)
            best= eval_acc['top1']
            thres = 0
        else:
            thres +=1
        if thres >= opt.est:
            print("early stopping triggered")
            break
            
        

def save_checkpoint(state, is_best, opt, filename='checkpoint.pth.tar'):
    print('saving checkpoint..')
    if is_best: torch.save(state, os.path.join(opt.save_path, 'model_best.pth.tar'))
    else: torch.save(state, os.path.join(opt.save_path, filename))

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
    
def accuracy_thresh(y_pred:Tensor, y_true:Tensor):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    top1, top3, top5 = 0,0,0
    y_pred = nn.Softmax()(y_pred)
    _, y_pred = torch.topk(y_pred, 5)
    y_pred = y_pred.float()
    y_true = y_true.reshape(y_pred.shape[0],1).float()
    for i in range(y_pred.shape[0]):
        if y_true[i] in y_pred[i]:
            top5 +=1
        if y_true[i] in y_pred[i][:3]:
            top3 +=1
        if y_true[i] in y_pred[i][:1]:
            top1 +=1
    return top5, top3, top1 
    
def eval(eval_examples, opt):
    eval_features = convert_examples_to_features(eval_examples, opt.max_seq_length, tokenizer)
    eval_data = SequenceImgDataset(eval_features)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", opt.eval_batch_size)
    # all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    # all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    # all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
    # all_img = torch.tensor([f.image for f in train_features], dtype=torch.float)
    # eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_img)
    
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=opt.eval_batch_size, num_workers=16)
    
    all_logits = None
    all_labels = None
    
    model.eval()
    eval_loss, eval_top5_accuracy, eval_top3_accuracy, eval_top1_accuracy = 0, 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids, img in eval_dataloader:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()
        img = img.permute(0,3,1,2)
        assert img.shape[1:] == (3,224,224)
        img = img.cuda()

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids, image=img)
            logits = model(input_ids, segment_ids, input_mask, image=img)

#         logits = logits.detach().cpu().numpy()
#         label_ids = label_ids.to('cpu').numpy()
#         tmp_eval_accuracy = accuracy(logits, label_ids)
        top5, top3, top1 = accuracy_thresh(logits, label_ids)
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
            
        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:    
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)
        

        eval_top5_accuracy += top5
        eval_top3_accuracy += top3
        eval_top1_accuracy += top1
        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_top5_accuracy = eval_top5_accuracy / nb_eval_examples
    eval_top3_accuracy = eval_top3_accuracy / nb_eval_examples
    eval_top1_accuracy = eval_top1_accuracy / nb_eval_examples
    
    result = {'eval_loss': eval_loss,
              'top5': eval_top5_accuracy,
              'top3': eval_top3_accuracy,
              'top1': eval_top1_accuracy}

    output_eval_file = os.path.join(opt.save_path, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        writer.write('\n=================')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result
    
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
    

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])    
        return model, optimizer
        
    return model

def model_ensemble(models):
    sds = [model.state_dict() for model in models]
    num_dicts = len(models)
    for name in sds[0]:
        for i in range(1,num_dicts):
            sds[0][name] += sds[i][name]
        sds[0][name] /= num_dicts


if __name__ == '__main__':
    opt = parser.parse_args()
    opt= set_default_opt(opt)
    print_options(opt)
    if opt.multilingual:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=opt.do_lower_case)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=opt.do_lower_case)
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    processor = MultiLabelTextProcessor(filename=opt.filename, num_classes=opt.num_classes, max_num=opt.downsample)
    if opt.weighted:
        opt.weighted = torch.tensor(processor.get_class_weights()).float().cuda()
        print('==============')
        print(opt.weighted)
        assert(len(opt.weighted) == opt.num_classes)
    model = BERT(opt, num_labels=opt.num_classes).cuda()
    train_examples = processor.get_train_examples()
    train_features = convert_examples_to_features(train_examples, opt.max_seq_length, tokenizer)
    eval_examples = processor.get_dev_examples()
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", opt.train_batch_size)
    train_data = SequenceImgDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opt.train_batch_size, num_workers=16)
    num_train_steps = int(len(train_examples) / opt.train_batch_size / opt.gradient_accumulation_steps * opt.epoch)
    t_total = num_train_steps
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.learning_rate,
                         warmup=opt.warmup_proportion,
                         t_total = t_total)

    if opt.resume is not None:
        model, optimizer = load_checkpoint(opt.save_path +'/'+ opt.resume, model, optimizer).cuda()
    fit(opt, train_dataloader, optimizer, eval_examples)