import sys
from unicodedata import category
import pandas as pd
import numpy as np
import pickle as pkl
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Normalizer, MaxAbsScaler 
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df[df.columns[3:]]
    y = y.drop(columns=['child_alone']) #all zeros
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in sw]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(n_jobs, cv):
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('norm', MaxAbsScaler()),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
    ], verbose=True
)
    parameters = {
    'norm':[Normalizer(), MaxAbsScaler()],
    'clf__estimator':[LogisticRegression(C=1, max_iter=1000), 
                      LogisticRegression(C=0.1, max_iter=1000),
                      LinearSVC(C=1), 
                      LinearSVC(C=0.1)]
}

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=n_jobs, cv=cv)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = category_names

    for i, c in enumerate(category_names):
        print("--------------------------------------------")
        print(c)
        print(classification_report(y_test[c], y_pred[:, i], zero_division=0))
        print("--------------------------------------------")


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pkl.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        n_jobs = -1
        cv = 5
        model = build_model(n_jobs=n_jobs, cv=cv)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()