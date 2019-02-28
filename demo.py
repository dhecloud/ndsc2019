from test_BERT import *
import torch
import torch.nn as nn
import time
from dataloaders import *
from dataset import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='ndsc_eval')
parser.add_argument('--last_layer_size', type=int, default=768, help='last layer size for resnet')
parser.add_argument('--batchsize', type=int, default=64, help='train batch size')

opt = parser.parse_args()

def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

    return model

if __name__ == '__main__':
    MAX_SEQ_LEN = 64
    test = pd.read_csv('data/test.csv', encoding='utf-8')
    sample = pd.read_csv('data/data_info_val_sample_submission.csv', encoding='utf-8')
    assert test.shape[0] == sample.shape[0]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BERT(opt).cuda()
    model = load_checkpoint('experiments/model_best.pth.tar', model).cuda()
    model.eval()
    softmax = nn.Softmax()
    processor = MultiLabelTextProcessor()
    test_examples = processor.get_test_examples()
    test_features = convert_examples_to_features(test_examples, MAX_SEQ_LEN, tokenizer)
    test_dataset = SequenceImgDataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=opt.batchsize, num_workers=4)
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
            sample.loc[list(range(i*opt.batchsize,(i*opt.batchsize)+no)),'Category'] = pred
                
            # print("time taken for one search: ", time.time()-stime, "seconds")
    
    sample.to_csv('submission.csv', index=False)