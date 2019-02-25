import ast
import logging
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
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm, trange
from torch.nn import BCEWithLogitsLoss
import pickle


import argparse
parser = argparse.ArgumentParser(description='ndsc')
parser.add_argument('--expand', action='store_true', help='expand dataset?')
parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
parser.add_argument('--batchsize', type=int, default=100, help='train batch size')

def fit(args, train_dataloader, optimizer, eval_examples):
    
    global_step = 0
    cur_epoch = 0
    model.train()
    for i_ in tqdm(range(int(args['num_train_epochs'])), desc="Epoch"):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
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
        # save_checkpoint(state, False, args, filename=str(cur_epoch)+"_checkpoint.pth.tar")
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
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])
    
    all_logits = None
    all_labels = None
    
    model.eval()
    eval_loss, eval_top5_accuracy, eval_top3_accuracy, eval_top1_accuracy = 0, 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
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
    
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
    
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError() 

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
        
    
class MultiLabelTextProcessor(DataProcessor):

    def __init__(self, data_dir='data/'):
        self.data_dir = data_dir
        self.labels = None
        filename = 'train.csv'
        self.data_df = pd.read_csv(os.path.join(self.data_dir, filename))
        print(self.data_df.shape)
        total_rows =  self.data_df.shape[0]
        print("total number of rows", total_rows) 
        shuffle= [i for i in range(total_rows)]
        random.shuffle(shuffle)
        train_idx = shuffle[:int(total_rows*0.8)]
        val_idx = shuffle[int(total_rows*0.8):]
        print('splitting df..')
        self.train_df = self.data_df.loc[train_idx,:].reset_index(drop=True)
        self.val_df = self.data_df.loc[val_idx,:].reset_index(drop=True)

    def get_train_examples(self, data_dir='data/', size=-1):
        if size == -1:
    #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            train_examples = self._create_examples(self.train_df, "train")
            return train_examples
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename))
    #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df.sample(size), "train")
        
    def get_dev_examples(self, data_dir='data/', size=-1):
        """See base class."""
        if size == -1:
    #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(self.val_df, "dev")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename))
    #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df.sample(size), "dev")

    def get_test_examples(self, data_dir, data_file_name, size=-1):
    #         data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
        if size == -1:
            return self._create_examples(self.test_df, "test")
        else:
            return self._create_examples(data_df.sample(size), "test")

    def get_labels(self):
        """See base class."""
        self.labels = [i for i in range(58)]
        
        return self.labels

    def _create_examples(self, df, set_type, labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(df.values):
            guid = row[0]
            text_a = row[1]
            if labels_available:
                labels = row[2]
            else:
                labels = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples
                
                        
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        

#         label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=float(example.labels)))
    return features
        
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()           
             

class BERT(nn.Module):
    def __init__(self, num_labels=58):
        super().__init__()
        
        self.num_labels= num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.long())
            return loss
        else:
            return logits


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
    processor = MultiLabelTextProcessor()
    train_examples = processor.get_train_examples()
    label_list = processor.get_labels()
    train_features = convert_examples_to_features(train_examples, label_list, args['max_seq_length'], tokenizer)
    eval_examples = processor.get_dev_examples()
        
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args['train_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
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
    
    

# tmp = []
# for i, q in enumerate(df):
#     length = len(q)
#     if length == 0:
#         pass
#     elif length > 50:
#         idxs = random.sample(range(length),50)
#     else:
#         idxs = range(length)
#     if length != 0:
#         df[i] = df[i][idxs]

# to trim 
# new=[]
# cur=0
# count = 0
# for i, cls in enumerate(b):
#     if cur != cls:
#         count=0
#         cls += 1
#     if count > 4: pass
#     elif cur == cls:
#         new.append(i)       
#         count += 1 
#     else:
#         cur += 1
#         count = 0