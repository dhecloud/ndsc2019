from test_BERT import *
import torch
import torch.nn as nn
import time

def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])


    return model
    
def normalizeStringInDF(s):
    s = s.str.normalize('NFC')
    s = s.str.replace(r"([.!?])", r" \1")
    s = s.str.replace(r"[^a-zA-Z0-9.!?]+", r" ")
    s = s.str.replace(r"<a.*</a>", 'url')
    return s
        
if __name__ == '__main__':
    MAX_SEQ_LEN = 64
    test = pd.read_csv('data/test.csv', encoding='utf-8')
    sample = pd.read_csv('data/data_info_val_sample_submission.csv', encoding='utf-8')
    assert test.shape[0] == sample.shape[0]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BERT().cuda()
    model = load_checkpoint('experiments/0_checkpoint.pth.tar', model).cuda()
    model.eval()
    softmax = nn.Softmax()
    with torch.no_grad():
        for i, strin in test.iterrows():
            stime = time.time()
            input_example = InputExample(guid=strin[0], text_a=strin[1])
            tokens = tokenizer.tokenize(input_example.text_a)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(tokens)
            padding = [0] * (MAX_SEQ_LEN - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == MAX_SEQ_LEN
            assert len(input_mask) == MAX_SEQ_LEN
            assert len(segment_ids) == MAX_SEQ_LEN
            labels_ids = []
            input_feat = InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=labels_ids)
            logits = model(torch.tensor(input_ids).unsqueeze(0).cuda(), torch.tensor(segment_ids).unsqueeze(0).cuda(), torch.tensor(input_mask).unsqueeze(0).cuda(), labels=None)
            pred = softmax(logits)
            _, topk = torch.topk(pred, 5)
            topk = topk[0].cpu().numpy()
            ans = topk[0]
            print('ans', ans)
            sample.loc['itemid','Category'] = ans
                
            print("time taken for one search: ", time.time()-stime, "seconds")
    
    sample.to_csv('submission.csv', index=False)
# 
# for i, val in adf.iterrows():
#     if (val[0][0] is "[") and (val[0][4] is not "]"):
#         tmp = ast.literal_eval(val[0])
#         for j, sent in enumerate(tmp):
#             idx.append(i)
#             str_list.append(sent)
#     else:
#         idx.append(i)
#         str_list.append(val[0])

def check(x):
    list1 = [[] for _ in range(100)]
    i = 0
    for l in x:
        l = l.strip()
        if l is '':
            pass
        elif l[-1] is "=":
            i += 1
        elif l[-1] is ":":
            start = l.find("'")
            end = l.rfind("'")
            l = l[start+1:end]
            l = nlp(l)
            # try:
                # print(l)
            list1[i].append(l)
            # except UnicodeEncodeError: continue
        else:
            l = nlp(l)
            # print(l)
            try:
                print(l)
                list1[i].append(l)
            except UnicodeEncodeError: continue
    for j in range(i-1 , -1, -1):
        print(j, str(list1[j][0]))
        with open("sep/"+ str(list1[j][0]).replace("/","-") + ".txt", 'w') as f:
            for sent in list1[j][1:]: #21
                f.write(str(sent) + "\n")
                
            # >> ’
# for file in files:
#     print(file)
#     with open(file, 'r', encoding='utf-8') as f:
#         a=f.readlines()
#         check(a)