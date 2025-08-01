import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re

count = CountVectorizer()

docs = np.array(['The sun is shining',
                'The weather is sweet',
                'The sun is shining and the weather is sweet',
                'and one and one is two'])

bag = count.fit_transform(docs)

df = df.reindex(np.random.permutation(df.index))

tfidf = TfidfTransformer(use_idf = True, norm = 'l2', smooth_idf = True)

np.set_printoptions(precision=2)

df.loc[0,'review'][-50:]

def preprocess_text(text):
    text = re.sub('<[^>]*>', '', text) # Remove HTML tags
    emoticons = re.findall('(?::|;|=) (?:-)?(?:\)|\(|D|P)', text)
    
    text = (re.sub('[\W]+', '', text.lower()) + ''.join(emoticons).replace('-',''))
    
    return text 

preprocess_text('is seven title brazil not available')

