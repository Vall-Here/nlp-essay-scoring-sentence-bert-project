def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', encoding='latin1')
    df = df[['essay_id', 'essay_set', 'essay', 'domain1_score']].dropna()
    return df


essays = load_data('../Dataset/asap-aes/training_set_rel3.tsv')

class EssayPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english') )
        self.lemmatizer = WordNetLemmatizer()
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        self.sbert = SentenceTransformer('all-mpnet-base-v2')
        
    def preprocess_text(self, text):
        # Basic
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenization and stopword removal
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmi
        pos_tags = nltk.pos_tag(tokens)
        lemmatized = []
        for word, tag in pos_tags:
            if tag.startswith('V'):  # Verb
                lemmatized.append(self.lemmatizer.lemmatize(word, 'v'))
            elif tag.startswith('J'):  # Adjective
                lemmatized.append(self.lemmatizer.lemmatize(word, 'a'))
            elif tag.startswith('R'):  # Adverb
                lemmatized.append(self.lemmatizer.lemmatize(word, 'r'))
            else:  # Noun
                lemmatized.append(self.lemmatizer.lemmatize(word))
                
        return ' '.join(lemmatized)
    
    def extract_linguistic_features(self, text):
         # Grammar and spelling
        matches = self.grammar_tool.check(text)
        grammar_errors = len(matches)
        
        # Readability
        readability = flesch_kincaid_grade(text)
        
        
        # Vocabulary richness
        words = word_tokenize(text)
        unique_words = set(words)
        vocab_richness = len(unique_words) / max(1, len(words))
        
        # Essay structure
        sentences = nltk.sent_tokenize(text)
        avg_sentence_len = sum(len(word_tokenize(s)) for s in sentences) / max(1, len(sentences))
        
        return {
            'grammar_errors': grammar_errors,
            'readability': readability,
            'vocab_richness': vocab_richness,
            'avg_sentence_len': avg_sentence_len
        }
    
    def get_sbert_embedding(self, text):
        return self.sbert.encode(text, show_progress_bar=False)


preprocessor = EssayPreprocessor()

essays['processed_text'] = essays['essay'].apply(preprocessor.preprocess_text)

linguistic_features = essays['processed_text'].apply(preprocessor.extract_linguistic_features)
linguistic_df =  pd.json_normalize(linguistic_features)


print("Generating SBERT embeddings...")
sbert_embeddings = np.array(essays['processed_text'].apply(preprocessor.get_sbert_embedding).tolist())
sbert_df = pd.DataFrame(sbert_embeddings)

X = pd.concat([linguistic_df, sbert_df], axis=1)
y = essays['domain1_score']


class AESModel(nn.Module):
    def __init__(self, input_dim):
        super(AESModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.relu(x)
        return self.fc_out(x)


def train_model(X_train, y_train, params):

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    model = AESModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    

    for epoch in range(params['epochs']):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    return model



def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test.values)
        predictions = model(X_test_tensor).squeeze().numpy()

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return {
        'predictions': predictions,
        'mse': mse,
        'rmse': rmse,
        'r2': r2  
    }


def objective(trial):
    
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'epochs': trial.suggest_int('epochs', 20, 100),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }
    
    kf = KFold(n_splits=5)
    rmse_scores = []  
    r2_scores = []  
    

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
   
        model = train_model(X_train, y_train, params)
      
        metrics = evaluate_model(model, X_val, y_val)
        
  
        rmse_scores.append(metrics['rmse'])
        r2_scores.append(metrics['r2'])

    return np.mean(rmse_scores,r2_scores)



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
best_params = study.best_params
print("Best hyperparameters:", best_params)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


final_model = train_model(X_train, y_train, best_params)

results = evaluate_model(final_model, X_test, y_test)