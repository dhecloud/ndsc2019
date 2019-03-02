import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='merge csv')
parser.add_argument('--beauty', type=str,  help='beauty')
parser.add_argument('--fashion', type=str, help='fashion')
parser.add_argument('--mobile', type=str, help='mobile')
opt = parser.parse_args()


beauty = pd.read_csv('experiments/'+opt.beauty+'/submission.csv', encoding='utf-8')
mobile = pd.read_csv('experiments/'+ opt.mobile + '/submission.csv', encoding='utf-8')
fashion = pd.read_csv('experiments/'+ opt.fashion +'/submission.csv', encoding='utf-8')

new = pd.concat([beauty,fashion,mobile])
assert(new.shape == (172402,2))
new.to_csv('data/submission.csv',index=False)