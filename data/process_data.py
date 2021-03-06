import sys
import pandas as pd
from sqlalchemy import create_engine
import IPython


def load_data(messages_filepath, categories_filepath):
    """
    Read csv files and merge them into a pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath, index_col=0)
    categories = pd.read_csv(categories_filepath, index_col=0)
    
    return messages.merge(categories, on='id')
    

def clean_data(df):
    """
    Prepare the data, dropping useless characters and converting formats
    """
    # print(df.head())
    categories = pd.DataFrame(df['categories'])
    messages = pd.DataFrame(df[['message', 'original', 'genre']])
    
    categories = categories.apply(lambda x: x.str.split(";", expand=True).iloc[0], axis=1)
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    for i, c in enumerate(categories['related']):
        if c==2:
            categories['related'].iloc[i] = 0 #messages with related=2 are in another language, are incomplete or incomprehensible 

    df = messages.merge(categories, on='id')
    IPython.embed()
    return df.drop_duplicates()


def save_data(df, database_filename):
    """
    Saves a DataFrame in a SQL database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()