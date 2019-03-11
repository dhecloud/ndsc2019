import pandas as pd
import nltk
nltk.download('stopwords')
import random
import argparse
from nltk.corpus import stopwords

parser = argparse.ArgumentParser(description='ndsc')
import argparse
parser.add_argument('--translated', action='store_true', help='translated')
parser.add_argument('--model', type=str, default='naivebayes',choices=['naivebayes', 'dtree'], help='choice of model')
opt = parser.parse_args()
import pickle

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

if opt.translated:
    fashion = pd.read_csv('data/fashion_translated_train.csv')
    experiment = 'translated_'+ opt.model
else:
    fashion = pd.read_csv('data/fashion_only_train.csv')
    experiment = 'original_'+ opt.model

print(experiment)
fashion_titles = [x[1] for x in list(fashion.title.items())]
fashion_titles = [nltk.word_tokenize(x) for x in fashion_titles]
fashion_words = []
for title in fashion_titles:
    fashion_words += title
fashion_cat = [x[1] for x in list(fashion.Category.items())]
assert(len(fashion_cat) == len(fashion_titles))
fashion_data = [(fashion_titles[i], fashion_cat[i]) for i in range(len(fashion_titles))]
random.shuffle(fashion_data)
all_words = nltk.FreqDist(w.lower() for w in fashion_words)
word_features = list(all_words)[:2000]
word_features = [x for x in word_features if x not in stopwords.words('english')]
word_features = [x for x in word_features if len(x) > 1]

featuresets = [(document_features(d),c) for (d,c) in fashion_data]
num_samples = len(featuresets)
train_set, test_set = featuresets[:int(0.8*num_samples)], featuresets[int(0.8*num_samples):]
if opt.model == 'naivebayes':
    print('training naive bayes')
    classifier = nltk.NaiveBayesClassifier.train(train_set)
else:
    print('training decision tree')
    classifier = nltk.DecisionTreeClassifier.train(train_set)
with open('data/'+experiment+ '.pkl', 'wb') as f:
    pickle.dump(classifier,f)  
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)