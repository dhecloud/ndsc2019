import pandas as pd
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import os
from PIL import Image
random.seed(99)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None, image= None):
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
        self.image = image

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, image):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.image = image
    
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

    def __init__(self, data_dir='data/', max_num=1000):
        self.data_dir = data_dir
        self.labels = None
        filename = 'train.csv'
        self.data_df = pd.read_csv(os.path.join(self.data_dir, filename))
        print(self.data_df.shape)
        keep = []
        for i in range(58):
            idxs = self.data_df.index[self.data_df.Category==i].tolist()
            if max_num > len(idxs):
                keep += random.sample(idxs,  len(idxs))
            else:
                keep += random.sample(idxs,  max_num)
        self.data_df = self.data_df.iloc[keep,:].reset_index(drop=True)
        total_rows =  self.data_df.shape[0]
        print("total number of rows", total_rows)

        shuffle= [i for i in range(total_rows)]
        random.shuffle(shuffle)
        train_idx = shuffle[:int(total_rows*0.8)]
        val_idx = shuffle[int(total_rows*0.8):]
        print('splitting df..')
        self.train_df = self.data_df.loc[train_idx,:].reset_index(drop=True)
        self.val_df = self.data_df.loc[val_idx,:].reset_index(drop=True)
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

    def get_train_examples(self, data_dir='data/', size=-1):
        if size == -1:
            train_examples = self._create_examples(self.train_df, "train")
            return train_examples
        else:
            return self._create_examples(self.train_df.sample(size), "train")
        
    def get_dev_examples(self, data_dir='data/', size=-1):
        """See base class."""
        if size == -1:
            return self._create_examples(self.val_df, "dev")
        else:
            return self._create_examples(self.val_df.sample(size), "dev")

    def get_test_examples(self, size=-1):
        if size == -1:
            return self._create_examples(self.test_df, "test",labels_available=False)
        else:
            return self._create_examples(self.test_df.sample(size), "test",labels_available=False)

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
                img_path = row[3]
            else:
                img_path = row[2]
            if not img_path.endswith('.jpg'):
                img_path = img_path +'.jpg'
            # image = np.array(Image.open('data/'+img_path).resize((224,224)))
            if labels_available:
                labels = row[2]
            else:
                labels = 999
            examples.append(
                InputExample(guid=guid, text_a=text_a, labels=labels, image=img_path))
        return examples
                
                        
def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""


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
                              label_ids=example.labels,
                              image = example.image))
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