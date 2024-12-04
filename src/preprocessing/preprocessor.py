import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

class PreProcessor:

    def __init__(self, data):
        self.data = data


    def split(self, data, train_ratio=0.6):
        '''This functions splits the dataset into a ratio appropriate
            train, eval and test split
        '''

        train_df, test_val_df = train_test_split(data, test_size=1-train_ratio, random_state=42)

        test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

        return train_df, test_df, val_df

    def preprocess(self, data, column):
        data[column] = data[column].apply(self.lower)
        data[column] = data[column].apply(self.strip_html)
        data[column] = data[column].apply(self.remove_chars)
        data[column] = data[column].apply(self.stopword_removal)
        data[column] = data[column].apply(self.stemming)

        return data



    def lower(self,text):
        return text.lower()

    def strip_html(self,text):
        '''This function was taken from:
        https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews
        '''

        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()


    def remove_chars(self, text):
        ''' This function removes special characters from the data
        '''

        pattern=r'[^a-zA-z0-9\s]'
        text=re.sub(pattern,'',text)
        return text


    def stemming(self, text):
        stemmer = PorterStemmer()
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        return text


    def stopword_removal(self, text):
        stopword_list=nltk.corpus.stopwords.words('english')
        tokenizer=ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)

        return filtered_text



        

