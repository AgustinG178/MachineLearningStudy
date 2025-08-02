import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
import pandas as pd
import pyprind
import os
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')

porter = PorterStemmer()
count = CountVectorizer()

basepath = 'training_data/aclImdb'

labels =  {'pos':1, 'neg':0}
pbar = pyprind.ProgBar(50000)
data = []
for s in ('test', 'train'):
    for l in ('pos','neg'):
        path = os.path.join(basepath,s,l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
                txt = infile.read()
            data.append([txt, labels[l]])
            pbar.update()

df = pd.DataFrame(data, columns=['review', 'sentiment'])

df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv('movie_data.csv', encoding='utf-8')

tfidf = TfidfTransformer(use_idf = True, norm = 'l2', smooth_idf = True)

np.set_printoptions(precision=2)

def preprocess_text(text):
    text = re.sub('<[^>]*>', '', text)  # Remove HTML tags
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')

    return text

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return[porter.stem(word) for word in text.split()]

stop = stopwords.words('english')

print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])



