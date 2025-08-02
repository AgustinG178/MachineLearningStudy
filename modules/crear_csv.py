import os
import pandas as pd
import nltk
import pyprind
from nltk.corpus import stopwords

pkl_path = 'training_data/movie_data.pkl'
csv_path = 'training_data/movie_data.csv'
basepath = 'training_data/aclImdb'

if not os.path.exists(pkl_path):
    nltk.download('stopwords')
    stop = stopwords.words('english')  
    labels = {'pos':1, 'neg':0}
    pbar = pyprind.ProgBar(50000)
    data = []

    for s in ('test', 'train'):
        for l in ('pos','neg'):
            path = os.path.join(basepath, s, l)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                    txt = infile.read()
                data.append([txt, labels[l]])
                pbar.update()

    df = pd.DataFrame(data, columns=['review', 'sentiment'])
    df.to_csv(csv_path, index=False, encoding='utf-8')
    df.to_pickle(pkl_path)  

df = pd.read_pickle(pkl_path)
