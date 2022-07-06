import pandas as pd
import re 
import joblib
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer, MaxAbsScaler 
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC


sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    """
    Loads a clean SQL database and return it as pandas Dataframes, ready to train
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df[df.columns[3:]]
    y = y.drop(columns=['child_alone']) #all zeros
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """
    Tokenize and lemmatize texts, returns clean lists
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in sw]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(n_jobs, cv):
    """
    Build a pipeline to prepare and train a dataset. Includes gridsearch
    """
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
    """
    Test the model and print results.
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = category_names

    for i, c in enumerate(category_names):
        print("--------------------------------------------")
        print(c)
        print(classification_report(y_test[c], y_pred[:, i], zero_division=0))
        print("--------------------------------------------")


def save_model(model, model_filepath):
    """
    Saves the trained model as a .pkl file
    """
    with open(model_filepath, 'wb') as file:
        joblib.dump(model, file)