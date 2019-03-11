def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np
import sys
import time
import os
import pandas as pd
import pickle
from torchtext import data
import random
from PIL import Image

class SequenceImgDataset(Dataset):
    def __init__(self, features):


        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        self.all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.float)
        self.all_img_pth = [f.image for f in features]
        self.centercrop = transforms.CenterCrop(150)
        # self.normalizer = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        assert self.all_input_ids.shape[0] == self.all_input_mask.shape[0] == self.all_segment_ids.shape[0] == self.all_label_ids.shape[0] ==len(self.all_img_pth)
        
    def __len__(self):
        return self.all_input_ids.shape[0]

    def __getitem__(self, i):
        #use_normalizer after all done training
        image = torch.tensor(np.array(self.centercrop(Image.open('data/'+self.all_img_pth[i]).convert('RGB').resize((224,224))).resize((224,224))) ,dtype=torch.float)
        return   self.all_input_ids[i], self.all_input_mask[i], self.all_segment_ids[i], self.all_label_ids[i], image