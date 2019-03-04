import copy
from test_BERT import *
import torch
import torch.nn as nn
import time
from dataloaders import *
from dataset import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='ndsc_eval')
parser.add_argument('--cat', type=str, default='beauty', help='last layer size for resnet')
parser.add_argument('--name', nargs='+', type=str, help='experiment name')
parser.add_argument('--last_layer_size', type=int, default=768, help='last layer size for resnet')
parser.add_argument('--batchsize', type=int, default=256, help='train batch size')
parser.add_argument('--no_bert', action='store_true', help='dont use bert')
parser.add_argument('--images', type=bool, default=True, help='dont use images to train')
parser.add_argument('--freeze_bert', action='store_true', help='freeze bert')
opt = parser.parse_args()
#beauty - classes 0-16 predicted - classes 0-16 in data
#fashion - classes 0-14 predicted - classes 17-30 in data
#mobile - classes 0-26 predicted - classes 31-57 in data
idxs = {'beauty': list(range(17)), 'fashion': list(range(17,31)), 'mobile': list(range(31,58)) }
opt.num_classes = len(idxs[opt.cat])

def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == '__main__':
    MAX_SEQ_LEN = 64
    test = pd.read_csv('data/'+opt.cat+'_test.csv', encoding='utf-8')
    test['Category'] = 99
    # sample = pd.read_csv('data/submission.csv', encoding='utf-8')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BERT(opt, num_labels=opt.num_classes).cuda()
    if len(opt.name) > 1:
        models = [copy.deepcopy(load_checkpoint('experiments/'+ name +'/model_best.pth.tar', model).cuda()) for name in opt.name]
        model_ensemble(models)
        model = copy.deepcopy(models[0])
        del models
        
    else:
        model = load_checkpoint('experiments/'+ opt.name[0] +'/model_best.pth.tar', model).cuda()
    model.eval()
    softmax = nn.Softmax()
    processor = MultiLabelTextProcessor(test=opt.cat+'_test.csv')
    test_examples = processor.get_test_examples()
    test_features = convert_examples_to_features(test_examples, MAX_SEQ_LEN, tokenizer)
    test_dataset = SequenceImgDataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=opt.batchsize, num_workers=8)
    with torch.no_grad():
        # for i, strin in test.iterrows():
        #     stime = time.time()
        #     input_example = InputExample(guid=strin[0], text_a=strin[1])
        #     tokens = tokenizer.tokenize(input_example.text_a)
        #     input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #     input_mask = [1] * len(input_ids)
        #     segment_ids = [0] * len(tokens)
        #     padding = [0] * (MAX_SEQ_LEN - len(input_ids))
        #     input_ids += padding
        #     input_mask += padding
        #     segment_ids += padding
        #     img_path = strin[2]
        #     if not img_path.endswith('.jpg'):
        #         img_path = img_path +'.jpg'
        #     assert len(input_ids) == MAX_SEQ_LEN
        #     assert len(input_mask) == MAX_SEQ_LEN
        #     assert len(segment_ids) == MAX_SEQ_LEN
        #     image = torch.tensor(np.array(Image.open('data/'+ img_path).convert('RGB').resize((224,224))) ,dtype=torch.float).unsqueeze(0).cuda()
        #     image = image.permute(0,3,1,2)
        #     assert image.shape[1:] == (3,224,224)
        #     labels_ids = []
        #     logits = model(torch.tensor(input_ids).unsqueeze(0).cuda(), torch.tensor(segment_ids).unsqueeze(0).cuda(), torch.tensor(input_mask).unsqueeze(0).cuda(), labels=None, image = image)
        print("test starting..")
        for i, (input_ids, input_mask, segment_ids, _, img) in enumerate(tqdm(test_dataloader)):
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            no = input_ids.shape[0]
            img = img.permute(0,3,1,2)
            assert img.shape[1:] == (3,224,224)
            img = img.cuda()
            
            logits = model(input_ids, segment_ids, input_mask, image=img)
            pred = softmax(logits)
            pred=  torch.argmax(pred, dim=1).cpu().numpy()
            test.loc[list(range(i*opt.batchsize,(i*opt.batchsize)+no)),'Category'] = pred
                
            # print("time taken for one search: ", time.time()-stime, "seconds")
    test = test.drop(['title', 'image_path'], axis='columns')
    test.Category = test.Category.replace(list(range(opt.num_classes)), idxs[opt.cat] )
    assert not (test.Category==99).any()
    if len(opt.name) > 1:
        test.to_csv('experiments/'+opt.cat+'_ensemble_submission.csv', index=False)
    else:
        test.to_csv('experiments/'+ opt.name[0]+'/submission.csv', index=False)
    # sample.update(other=test)
    # sample =sample.astype('int64')
    # sample.to_csv('data/submission.csv', index=False)