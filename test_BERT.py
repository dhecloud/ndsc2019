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

parser = argparse.ArgumentParser(description='ndsc')
parser.add_argument('--expand', action='store_true', help='expand dataset?')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--batchsize', type=int, default=32, help='train batch size')
parser.add_argument('--est', type=int, default=3, help='train batch size')

def fit(args, train_dataloader, optimizer, eval_examples):
    
    global_step = 0
    cur_epoch = 0
    model.train()
    for i_ in tqdm(range(int(args['num_train_epochs'])), desc="Epoch"):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids, img = batch
            img = img.permute(0,3,1,2)
            assert img.shape[1:] == (3,224,224)
            loss = model(input_ids, segment_ids, input_mask, label_ids, image=img)
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
    #             scheduler.batch_step()
                # modify learning rate with special warm up BERT uses
                lr_this_step = args['learning_rate'] * warmup_linear(global_step/t_total, args['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            state = {
                'arch': "QAC",
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }


        logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
        logger.info('Eval after epoc {}'.format(i_+1))
        save_checkpoint(state, False, args, filename=str(cur_epoch)+"_checkpoint.pth.tar")
        cur_epoch += 1 
        eval(eval_examples)
        

def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    print('saving checkpoint..')
    torch.save(state, os.path.join('experiments/', filename))
    if is_best:
        torch.save(state, os.path.join('experiments/', 'model_best.pth.tar'))

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
    
def accuracy_thresh(y_pred:Tensor, y_true:Tensor):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    top1,top3,top5 = 0,0,0
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
    
def eval(eval_examples):


    eval_features = convert_examples_to_features(eval_examples, label_list, args['max_seq_length'], tokenizer)
    eval_data = SequenceImgDataset(eval_features)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    # all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    # all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    # all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
    # all_img = torch.tensor([f.image for f in train_features], dtype=torch.float)
    # eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_img)
    
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])
    
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
            logits = model(input_ids, segment_ids, input_mask)

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

    output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
#             writer.write("%s = %s\n" % (key, str(result[key])))
    return result
    
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
    

def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])


    return model
    
if __name__ == '__main__':
    opt = parser.parse_args()
    args = {
    "train_size": -1,
    "val_size": -1,
    "task_name": "bb_qa",
    "no_cuda": False,
    "output_dir": 'experiments/output',
    "max_seq_length": 64,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": opt.batchsize,
    "eval_batch_size": opt.batchsize,
    "learning_rate": 3e-5,
    "num_train_epochs": opt.epoch,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
    }
    model = BERT().cuda()
    # model = load_checkpoint('experiments/0_checkpoint.pth.tar', model).cuda()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=args['do_lower_case'])
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    
    if os.path.isfile('data/train_stuff.pkl'):
        with open('data/train_stuff.pkl', 'rb') as p:
            print('data/train_stuff.pkl found!')
            processor, train_examples, label_list, train_features, eval_examples= pickle.load(p)
            print('data/train_stuff.pkl loaded!')
    else:
        processor = MultiLabelTextProcessor()
        train_examples = processor.get_train_examples()
        label_list = processor.get_labels()
        train_features = convert_examples_to_features(train_examples, label_list, args['max_seq_length'], tokenizer)
        eval_examples = processor.get_dev_examples()
        with open('data/train_stuff.pkl', 'wb') as f:
            pickle.dump([processor, train_examples, label_list, train_features, eval_examples],f)  
            print('data/train_stuff.pkl! saved')
        
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args['train_batch_size'])
    train_data = SequenceImgDataset(train_features)
    # all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    # all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    # all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
    # all_img_pth = torch.tensor([f.image for f in train_features], dtype=torch.uint8)
    # train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_img_pth)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'], num_workers=0)
    num_train_steps = int(len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])
    t_total = num_train_steps
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args['learning_rate'],
                         warmup=args['warmup_proportion'],
                         t_total=t_total)
    fit(args, train_dataloader, optimizer, eval_examples)