import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
import random

# WARNING! Need to download if not
#import nltk 
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Word Vectorization
from sklearn.feature_extraction.text import CountVectorizer

class SentimentDataset(): 
    def __init__(self, data_path: str, train_size: int, vectorizer: CountVectorizer = None, max_feature: int = 1000, is_train: bool = False):
        # check if everything is fine
        self.initialized = False
        
        self.is_train = is_train
        self.max_feature_size = max_feature
        print(f"data => vectorized: {vectorizer}")
        self.vectorizer = CountVectorizer(max_features=self.max_feature_size) if vectorizer is None else vectorizer
        
        self.data = pd.read_csv(data_path)
        self.data_size = self.data.size
        self.train_size = train_size if train_size < self.data_size and train_size > 0 else self.data_size
        
        # Shuffle the dataset to randomize the order of reviews
        # self.data = self.data.sample(frac=1, random_state=random.randint(0, 9999)).reset_index(drop=True)

        self.messages = self.data[self.data.columns[0]].tolist()[:self.train_size]
        self.labels = self.data[self.data.columns[1]][:self.train_size]
        self.class_size = len(self.labels.value_counts())        
        self.labels = self.labels.to_numpy()
        
        # preprocess textual data
        self.X_tokenized = self.preprocess_message() 
        self.X = self.vectorize_messages(self.X_tokenized)
        #print(f"self.X: {self.X}, and its type: {type(self.X)} - shape: {self.X.shape}")
        #print(f"self.labels: {self.labels}, and its type: {type(self.labels)} - shape: {self.labels.shape}")
        
        # everything is fine
        self.initialized = True
        return
    
    def __len__(self):
        return self.data_size
    
    def __str__(self):
        return f"SentimentDataset({self.data.column})"
    
    def check_init(self):
        if not self.initialized:
            raise RuntimeError("Dataset not initialized!")
        return True
    
    def get_class_size(self):
        self.check_init()
        return self.class_size
    
    def get_input_size(self):
        self.check_init()
        return self.X.shape[1]
    
    def getData(self):
        self.check_init()
        return self.X, self.labels
    
    def vectorize_messages(self, messages):
        # Fit and transform the preprocessed sentences
        if self.is_train:
            self.vectorizer.fit(messages)    
        messages = self.vectorizer.transform(messages).toarray()
        return messages
    
    def preprocess_message(self):
        # data holder for preprocessed_messages
        local_preprocessed_messages = []
        for message in tqdm(self.messages, desc="Data Preprocess"):
            # Remove special characters and digits
            text = re.sub('[^a-zA-Z]', ' ', message)
            
            # Convert to lowercase
            text = text.lower()
            
            # Tokenize the text
            tokens = word_tokenize(text)
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            stop_words.remove("no")
            stop_words.remove("not")
            tokens = [word for word in tokens if word not in stop_words]
            
            # Lemmatize the words
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            # Join the processed tokens back into a single string
            processed_text = ' '.join(tokens)
            local_preprocessed_messages.append(processed_text)
    
        return local_preprocessed_messages
    
    