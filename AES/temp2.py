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


sbert_linguistic = pd.concat([sbert_df_jawaban_siswa, linguistic_df], axis=1)
kunci_linguistic = pd.concat([sbert_df_kunci_jawaban, linguistic_df_kunci_jawaban], axis=1)

from scipy.spatial.distance import cosine
def calculate_similarity(row):
    student_emb = jawaban_embeddings[row.name]
    key_emb = kunci_embeddings[row['Kode'] - 1]
    return 1 - cosine(student_emb, key_emb)

df_merged['semantic_similarity'] = df_merged.apply(calculate_similarity, axis=1)

X = pd.concat([sbert_linguistic, df_merged[['semantic_similarity']]], axis=1).values
y = df_merged['Nilai'].values



# Split data 80% train, 20% test
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, np.arange(len(X)), 
    test_size=0.2, 
    random_state=42, 
    stratify=df_merged['Kode']
)

class AESModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super(AESModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.output_layer(x)) * 4  

def train_model(X_train, y_train, params, X_val=None, y_val=None):
    # Konversi ke tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    
    # Inisialisasi model
    model = AESModel(
        input_dim=X_train.shape[1],
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(params['device'])
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(params['device']), batch_y.to(params['device'])
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_X = torch.FloatTensor(X_val).to(params['device'])
                val_y = torch.FloatTensor(y_val).to(params['device'])
                val_outputs = model(val_X)
                val_loss = criterion(val_outputs.squeeze(), val_y).item()
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), './models/best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(f"Epoch {epoch+1}/{params['epochs']} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{params['epochs']} - Train Loss: {avg_train_loss:.4f}")
    
    if X_val is not None:
        model.load_state_dict(torch.load('./models/best_model.pt'))
    
    return model

def evaluate_model(model, X_test, y_test, device='cpu'):
    model.eval()
    with torch.no_grad():
        test_X = torch.FloatTensor(X_test).to(device)
        test_y = torch.FloatTensor(y_test).to(device)
        predictions = model(test_X).squeeze().cpu().numpy()
    
    # Hitung metrik evaluasi
    mse = mean_squared_error(y_test, predictions)
    qwk = cohen_kappa_score(
        (y_test * 4).round().astype(int), 
        (predictions * 4).round().astype(int), 
        weights='quadratic'
    )
    
    r2 = r2_score(y_test, predictions)
    
    print(f"\nEvaluation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"QWK: {qwk:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=predictions, scatter_kws={'alpha':0.3})
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Actual vs Predicted Scores')
    plt.show()
    
    return {'mse': mse, 'qwk': qwk, 'r2': r2}

def objective(trial):
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
        'num_layers': trial.suggest_int('num_layers', 2, 5),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    qwk_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = train_model(X_tr, y_tr, params, X_val, y_val)

        # model.eval()
        with torch.no_grad():
            val_X = torch.FloatTensor(X_val).to(params['device'])
            val_y = torch.FloatTensor(y_val).to(params['device'])
            val_outputs = model(val_X).squeeze().cpu().numpy()

        qwk = cohen_kappa_score(
            (val_y.cpu().numpy() * 4).round().astype(int),
            (val_outputs * 4).round().astype(int),
            weights='quadratic'
        )
        qwk_scores.append(qwk)

    return np.mean(qwk_scores)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, n_jobs=1)

best_params = study.best_params
best_params.update({
    'epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})

final_model = train_model(X_train, y_train, best_params, X_test, y_test)