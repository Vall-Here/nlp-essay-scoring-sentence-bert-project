import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report,f1_score
import numpy as np
from sentence_transformers import SentenceTransformer # type: ignore
import optuna
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

train_a = pd.read_csv('../Dataset/Data_A/data_train_A.csv')
dev_a = pd.read_csv('../Dataset/Data_A/data_dev_A.csv')
test_a = pd.read_csv('../Dataset/Data_A/data_test_A.csv')
train_b = pd.read_csv('../Dataset/Data_B/data_train_B.csv')
dev_b = pd.read_csv('../Dataset/Data_B/data_dev_B.csv')
test_b = pd.read_csv('../Dataset/Data_B/data_test_B.csv')


stimulus_a = ["Pemanasan global terjadi karena peningkatan produksi karbon dioksida yang dihasilkan oleh pembakaran fosil dan konsumsi bahan bakar yang tinggi.",
"Salah satu akibat adalah mencairnya es abadi di kutub utara dan selatan yang menimbulkan naiknya ketinggian air laut.",
"kenaikan air laut akan terjadi terus menerus meskipun dalam hitungan centimeter akan mengakibatkan perubahan yang signifikan.",
"Film “Waterworld”, adalah film fiksi ilmiah yang menunjukkan akibat adanya pemanasan global yang sangat besar sehingga menyebabkan bumi menjadi tertutup oleh lautan.",
"Negara-negara dan daratan yang dulunya kering menjadi tengelamn karena terjadi kenaikan permukaan air laut.",
"Penduduk yang dulunya bisa berkehidupan bebas menjadi terpaksa mengungsi ke daratan yang lebih tinggi atau tinggal diatas air.",
"Apa yang akan menjadi tantangan bagi suatu penduduk ketika terjadi situasi daratan tidak dapat ditinggali kembali karena tengelam oleh naiknya air laut."]

stimulus_b = ["Sebuah toko baju berkonsep self-service menawarkan promosi dua buah baju bertema tahun baru seharga Rp50.000,00. sebelum baju bertema tahun baru dibagikan kepada pembeli, sebuah layar akan menampilkan tampilan gambar yang menampilkan kondisi kerja di dalam sebuah pabrik konveksi/pembuatan baju. ",
"Kemudian pembeli diberi program pilihan untuk menyelesaikan pembeliannya atau menyumpangkan Rp50.000,00 untuk dijadikan donasi pembagian baju musim dingin di suatu daerah yang membutuhkan.",
"Delapan dari sepuluh pembeli memilih untuk memberikan donasi.",
"Menurut anda mengapa banyak dari pembeli yang memilih berdonasi?"]

stimulus_a_text = " ".join(stimulus_a)
stimulus_b_text = " ".join(stimulus_b)

for df in [train_a, dev_a, test_a]:
    df["TEXT"] = stimulus_a_text + " [SEP] " + df["RESPONSE"]

for df in [train_b, dev_b, test_b]:
    df["TEXT"] = stimulus_b_text + " [SEP] " + df["RESPONSE"]

stopwords_ukara = {'yang', 'lebih', 'untuk', 'akan', 'mereka', 'dan'}

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_ukara]
    return " ".join(tokens)


train_a["clean_response"] = train_a["RESPONSE"].apply(preprocess)
train_b["clean_response"] = train_b["RESPONSE"].apply(preprocess)
test_a["clean_response"] = test_a["RESPONSE"].apply(preprocess)
test_b["clean_response"] = test_b["RESPONSE"].apply(preprocess)
dev_a["clean_response"] = dev_a["RESPONSE"].apply(preprocess)
dev_b["clean_response"] = dev_b["RESPONSE"].apply(preprocess)


try:
    sbert_model
except NameError:
    sbert_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')


def extract_features(df, sbert_model):
    embeddings = sbert_model.encode(df['clean_response'].tolist())
    labels = df['LABEL'].values
    return embeddings, labels


train_embeddings_a, train_labels_a = extract_features(train_a, sbert_model)
dev_embeddings_a, dev_labels_a = extract_features(dev_a, sbert_model)
test_embeddings_a, test_labels_a = extract_features(test_a, sbert_model)

train_embeddings_b, train_labels_b = extract_features(train_b, sbert_model)
dev_embeddings_b, dev_labels_b = extract_features(dev_b, sbert_model)
test_embeddings_b, test_labels_b = extract_features(test_b, sbert_model)

class AESModel(nn.Module):
    def __init__(self, input_dim):
        super(AESModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model_with_params(params, train_embeddings, train_labels):
    if params['use_smote']:
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(train_embeddings, train_labels)
    else:
        X_train, y_train = train_embeddings, train_labels
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    
    model = AESModel(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    
    current_batch_size = params['batch_size']
    for epoch in range(params['epochs']):
        if params['batch_increase'] > 0 and (epoch+1) % params['increase_freq'] == 0:
            current_batch_size = min(current_batch_size + params['batch_increase'], len(train_dataset))
            
        train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return model

def objective(trial, train_embeddings, train_labels):
    params = {
        'epochs': trial.suggest_int('epochs', 10, 55),
        'batch_size': trial.suggest_int('batch_size', 2, 8),
        'batch_increase': trial.suggest_int('batch_increase', 0, 4),
        'increase_freq': trial.suggest_int('increase_freq', 2, 4),
        'use_smote': trial.suggest_categorical('use_smote', [True, False])
    }
    
    kf = KFold(n_splits=5)
    scores = []
    
    for train_idx, val_idx in kf.split(train_embeddings):
        X_train, X_val = train_embeddings[train_idx], train_embeddings[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        
        model = train_model_with_params(params, X_train, y_train)
        
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        scores.append(correct / total)
    
    return np.mean(scores)

def evaluate_model(model, embeddings, labels):
    dataset = TensorDataset(torch.FloatTensor(embeddings), torch.LongTensor(labels))
    loader = DataLoader(dataset, batch_size=32)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Incorrect', 'Correct']))
    
    return {
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, train_embeddings_a, train_labels_a), n_trials=100)
best_params = study.best_params
print("Best hyperparameters:", best_params)

final_model = train_model_with_params(best_params, train_embeddings_a, train_labels_a)

print("\nEvaluation on Validation Set:")
val_metrics = evaluate_model(final_model, dev_embeddings_a, dev_labels_a)

print("\nEvaluation on Test Set:")
test_metrics = evaluate_model(final_model, test_embeddings_a, test_labels_a)

print("\nFinal Metrics:")
print(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
print(f"Test - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, train_embeddings_b, train_labels_b), n_trials=100)


best_params = study.best_params
print("Best hyperparameters:", best_params)

final_model = train_model_with_params(best_params, train_embeddings_b, train_labels_b)

print("\nEvaluation on Validation Set:")
val_metrics = evaluate_model(final_model, dev_embeddings_b, dev_labels_b)

print("\nEvaluation on Test Set:")
test_metrics = evaluate_model(final_model, test_embeddings_b, test_labels_b)

print("\nFinal Metrics:")
print(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
print(f"Test - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")