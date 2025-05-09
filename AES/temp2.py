import pandas as pd
import numpy as np
import re
import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, cohen_kappa_score, r2_score
from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import optuna

import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = os.path.abspath('../Dataset/dataset mamasmasmudi.xlsx')



def load_data(dataset_path):
    df_dpk_tkj = pd.read_excel(dataset_path, sheet_name='DPK (TKJ)')
    df_mpp_rpl = pd.read_excel(dataset_path, sheet_name='MPP (RPL)')
    df_mpp_ppl = pd.read_excel(dataset_path, sheet_name='MPP (PPL) 2')
    df_mpp_tkj = pd.read_excel(dataset_path, sheet_name='MPP (TKJ-Telkom)')
    df_kunci_jawaban = pd.read_excel(dataset_path, sheet_name='Kunci Jawaban')
    
    df = pd.concat([df_dpk_tkj, df_mpp_rpl, df_mpp_ppl, df_mpp_tkj])
    
    return df, df_kunci_jawaban


essays, jawaban_essay = load_data(dataset_path)



class EssayPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('indonesian'))
        self.sbert = SentenceTransformer('firqaaa/indo-sentence-bert-base',
                                            device="cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        # text = re.sub(r'\d+', '', text) 

        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)

    def extract_linguistic_features(self, text):
        if not text:
            return {}

        words = word_tokenize(text)
        unique_words = set(words)
        vocab_richness = len(unique_words) / max(1, len(words))

        sentences = sent_tokenize(text)
        avg_sentence_len = sum(len(word_tokenize(s)) for s in sentences) / max(1, len(sentences))


        return {
            'vocab_richness': vocab_richness,
            'avg_sentence_len': avg_sentence_len,
            'total_words': len(words),

        }
        

    def minmaxnormalize_nilai(self, nilai):
        if not isinstance(nilai, (int, float)):
            return 0.0
        min_val = 0
        max_val = 4
        normalized_nilai = (nilai - min_val) / (max_val - min_val)
        return normalized_nilai
    
    def get_sbert_embedding(self, text):
        return self.sbert.encode(text, show_progress_bar=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['sbert']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sbert = SentenceTransformer('firqaaa/indo-sentence-bert-base',
                                            device="cuda" if torch.cuda.is_available() else "cpu")


preprocessor = EssayPreprocessor()

essays['processed_text'] = essays['Jawaban'].apply(preprocessor.preprocess_text)
essays['Nilai'] = essays['Nilai'].apply(preprocessor.minmaxnormalize_nilai)
jawaban_essay['processed_kunci_jawaban'] = jawaban_essay['Jawaban'].apply(preprocessor.preprocess_text)



linguistic_features = essays['processed_text'].apply(preprocessor.extract_linguistic_features)
linguistic_features_kunci_jawaban = jawaban_essay['processed_kunci_jawaban'].apply(preprocessor.extract_linguistic_features)
linguistic_df =  pd.json_normalize(linguistic_features)
linguistic_df_kunci_jawaban =  pd.json_normalize(linguistic_features_kunci_jawaban)


df_merged = pd.merge(essays, jawaban_essay,on="Kode")
df_merged = df_merged.rename(columns={'Pertanyaan_x': 'Pertanyaan', 'Jawaban_x': 'Jawaban siswa' , 'Jawaban_y': 'Kunci Jawaban'})
df_merged = df_merged.drop(columns=['Pertanyaan_y', 'Nama', 'Kelas_x', 'Kelas_y', 'jurusan', 'Mata Pelajaran'])

print("Generating SBERT embeddings...")
jawaban_embeddings = np.array(essays['processed_text'].apply(preprocessor.get_sbert_embedding).tolist())
sbert_df_jawaban_siswa = pd.DataFrame(jawaban_embeddings)
kunci_embeddings = np.array(jawaban_essay['processed_kunci_jawaban'].apply(preprocessor.get_sbert_embedding).tolist())
sbert_df_kunci_jawaban = pd.DataFrame(kunci_embeddings)