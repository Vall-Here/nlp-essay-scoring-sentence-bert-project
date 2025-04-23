import os
import pathlib
from collections import Counter
from statistics import mean

import re
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns; 
import textstat as ts

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from nltk.tokenize import sent_tokenize

import textstat
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel


dataset_dir_path =  pathlib.Path("../Dataset/asap-aes").resolve()
dataset_dir_path


raw_dataset = pd.read_csv(dataset_dir_path/'training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')

dataset = raw_dataset.copy()

dataset

dataset.essay_set.value_counts().rename_axis('essay_set').reset_index(name='counts').sort_values(
    'essay_set').plot(kind='bar', x="essay_set", y="counts");

df = dataset.dropna(axis='columns')

essays = df[['essay_id', 'essay_set', 'essay', 'domain1_score']].dropna()
essays

train_df, test_df = train_test_split(essays, test_size=0.2, random_state=42)


def preprocess_essay(text):
    text = text.lower()    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

train_df['processed'] = train_df['essay'].apply(preprocess_essay)
test_df['processed'] = test_df['essay'].apply(preprocess_essay)

test_df['processed']

try:
    model
except NameError:
    model = SentenceTransformer('all-mpnet-base-v2')
    
    
    
    

def get_sbert_embeddings(texts, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

print("Generating train embeddings...")
train_embeddings = get_sbert_embeddings(train_df['processed'].tolist())

print("Generating test embeddings...")
test_embeddings = get_sbert_embeddings(test_df['processed'].tolist())


# Convert to DataFrames
X_train = pd.DataFrame(train_embeddings)
X_test = pd.DataFrame(test_embeddings)
y_train = train_df['domain1_score']
y_test = test_df['domain1_score']


from xgboost import XGBRegressor
lwxgb = XGBRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42
)

# Train model
print("Training LwXGBoost...")
lwxgb.fit(X_train, y_train)

# Predict
predictions = lwxgb.predict(X_test)


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred.round(), weights='quadratic')

# Calculate metrics
qwk = quadratic_weighted_kappa(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("\nEvaluation Results:")
print(f"Quadratic Weighted Kappa (QWK): {qwk:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")