import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from primer_tokenizador import tokenizer, tokenizer_porter
from nltk.corpus import stopwords
import nltk
import os

nltk.download('stopwords')
stop = stopwords.words('english')
if os.path.exists('movie_data.pkl'):
    df = pd.read_pickle('movie_data.pkl')
else:
    df = pd.read_csv('movie_data.csv', encoding='utf-8')

X_train_full = df.loc[:25000, 'review'].values
Y_train_full = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
Y_test = df.loc[25000:, 'sentiment'].values

X_train, _, Y_train, _ = train_test_split(
    X_train_full, Y_train_full, train_size=0.05, stratify=Y_train_full, random_state=0
)

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [stop],
        'vect__tokenizer': [tokenizer_porter],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [1.0, 10.0]
    }
]

lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(random_state=0, solver='liblinear'))
])

gs_lr_tfidf = GridSearchCV(
    lr_tfidf,
    param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1
)

gs_lr_tfidf.fit(X_train, Y_train)

print(f'CV accuracy (50% training): {gs_lr_tfidf.best_score_:.3f}')
print(f'Test accuracy (full test set): {gs_lr_tfidf.score(X_test, Y_test):.3f}')
