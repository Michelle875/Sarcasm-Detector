#NOTE: NEED SKLEARN VERSION 1.70



import pandas as pd
data = input("input relative path to new data file (must have columns 'text' and 'label'): ")
data = pd.read_csv(data)
original_data = data.copy()

from scipy.sparse import load_npz
from joblib import load

#libraries
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
df_train = pd.read_csv("train.csv")
data['text'] = data['text'].str.strip()
df_train['text'] = df_train['text'].str.strip()

from nltk.tokenize import word_tokenize
data['text'] = data['text'].apply(word_tokenize)
df_train['text'] = df_train['text'].apply(word_tokenize)


nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

# Convert POS tags for WordNet
def get_wordnet_pos(tag):
    if tag.startswith('J'): 
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN   # default

def lemmatize_tokens(tokens):
    pos_tags = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
            for word, tag in pos_tags]

data['text'] = data['text'].apply(lemmatize_tokens)
df_train['text'] = df_train['text'].apply(lemmatize_tokens)

import pandas as pd
import numpy as np
import string
import re
from collections import Counter
import ast

# print(data['text'])
data["tokens"] = data["text"]
#.apply(ast.literal_eval)
df_train["tokens"] = df_train["text"]

df_train_str = pd.read_csv("train.csv")

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
# vectorizer = joblib.load("tfidf_vectorizer.pkl")

vectorizer = load("tfidf_vectorizer.pkl")

train_tfidf = load_npz("train_tfidf.npz")
data_tfidf = vectorizer.transform(original_data['text'])




import ast
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet = True)

sia = SentimentIntensityAnalyzer()

df_train['sentiment'] = df_train_str['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df_train['length_words'] = df_train_str['text'].apply(lambda x: len(x.split()))

data['sentiment'] = original_data['text'].apply(lambda x: sia.polarity_scores(x)['compound'])

data['length_words'] = original_data['text'].apply(lambda x: len(x.split()))


def extract_punctuation_features(text):
    features = {}
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['ellipsis_count'] = len(re.findall(r'\.{2,}', text))  # Two or more dots
    features['quote_count'] = text.count('"') + text.count("'")
    features['comma_count'] = text.count(',')
    features['period_count'] = text.count('.')
    features['semicolon_count'] = text.count(';')
    features['colon_count'] = text.count(':')
    features['dash_count'] = text.count('-') + text.count('—')
    features['multiple_exclamation'] = len(re.findall(r'!{2,}', text))
    features['multiple_question'] = len(re.findall(r'\?{2,}', text))
    features['mixed_punctuation'] = len(re.findall(r'(\?!|!\?)', text))
    total_punct = sum([features['exclamation_count'], features['question_count'],features['ellipsis_count'],features['quote_count'],features['semicolon_count'],features['colon_count'],
                       features['comma_count'], features['period_count'],features['dash_count']])
    text_length = len(text.split())
    features['punct_density'] = total_punct / max(text_length, 1)
    features['all_caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text))
    features['punct_types_used'] = len(set(c for c in text if c in string.punctuation))
    
    return features

def create_punctuation_features(text_series):
    punct_features = text_series.apply(extract_punctuation_features) 
    return pd.DataFrame(punct_features.tolist())


data_punct = create_punctuation_features(original_data['text'])

data = pd.concat([data, data_punct], axis=1)

df_tain_punct = create_punctuation_features(df_train_str['text'])

df_train = pd.concat([df_train, df_tain_punct], axis=1)

def parse_token_list(text_str):
    try:
        return ast.literal_eval(text_str)
    except:
        return []
    
def get_first_last_words(df):
    tokens = df['text'].apply(parse_token_list)
    
    first_words = tokens.apply(lambda x: x[0] if len(x) > 0 else '<EMPTY>')
    last_words = tokens.apply(lambda x: x[-1] if len(x) > 0 else '<EMPTY>')
    
    return first_words, last_words

data_first, data_last = get_first_last_words(data)

df_train_first, df_train_last = get_first_last_words(df_train)

def build_position_vocabulary(words_series, top_n=200, min_freq=5):
    
    word_counts = Counter(words_series)
    
    filtered_words = {word: count for word, count in word_counts.items() 
                     if count >= min_freq}
    
    vocab = [word for word, _ in sorted(filtered_words.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:top_n]]
    
    return vocab

first_word_vocab = build_position_vocabulary(data_first, top_n=200, min_freq=5)
last_word_vocab = build_position_vocabulary(data_last, top_n=200, min_freq=5)

df_train_first_vocab = build_position_vocabulary(df_train_first, top_n=200, min_freq=5)
df_train_last_vocab = build_position_vocabulary(df_train_last, top_n=200, min_freq=5)

def create_position_features(first_words, last_words, first_vocab, last_vocab):
    
    features = {}
    
    for word in first_vocab:
        features[f'first_{word}'] = (first_words == word).astype(int)
    
    for word in last_vocab:
        features[f'last_{word}'] = (last_words == word).astype(int)
    
    return pd.DataFrame(features)

data_position = create_position_features(data_first, data_last, first_word_vocab, last_word_vocab)
# data = pd.concat([data, data_position], axis=1)

# data.to_csv('test_features.csv', index=False)
df_train_position = create_position_features(df_train_first, df_train_last, df_train_first_vocab, df_train_last_vocab)


from sklearn.feature_extraction.text import CountVectorizer

def identity_tokenizer(tokens):
    return tokens  

vectorizer = CountVectorizer(
    tokenizer=identity_tokenizer,
    preprocessor=lambda x: x, 
    ngram_range=(1,2),
    lowercase=False
)

import ast

# df_train = pd.read_csv('train.csv')
X_train_bow = vectorizer.fit_transform(df_train['text'])
X_data_bow = vectorizer.transform(original_data['text'])
word_count = np.asarray(X_data_bow.sum(axis=0)).flatten()
features = vectorizer.get_feature_names_out()
freq_list = list(zip(features, word_count))
freq_list.sort(key=lambda x: x[1], reverse=True)









import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
def get_wordnet_pos(tag):
    if tag.startswith('J'): 
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN   # default

def tokens_to_pos_df(df, token_col='tokens'):
    words = []
    pos_tags = []

    for tokens in df[token_col]:
        tagged = nltk.pos_tag(tokens)   # list of (word, pos)
        for word, pos in tagged:
            words.append(word)
            pos_tags.append(pos)

    return pd.DataFrame({'word': words, 'pos': pos_tags})


pos_df = tokens_to_pos_df(df_train, 'text')

pos_data = tokens_to_pos_df(data, 'text')


pos_mapping = {
    # Nouns
    'NN': 'N', 'NNS': 'N', 'NNP': 'N', 'NNPS': 'N', 
    'PRP': 'N', 'PRP$': 'N', 'WP': 'N', 'WP$': 'N', 'EX': 'N',
    
    # Verbs
    'VB': 'V', 'VBD': 'V', 'VBG': 'V', 'VBN': 'V', 'VBP': 'V', 'VBZ': 'V',
    
    # Adjectives / Adverbs
    'JJ': 'A', 'JJR': 'A', 'JJS': 'A', 
    'RB': 'A', 'RBR': 'A', 'RBS': 'A', 'WRB': 'A',
    
    # Prepositions / Determiners / Modals / Conjunctions
    'IN': 'P', 'TO': 'P', 'DT': 'P', 'PDT': 'P', 'CC': 'P', 'MD': 'P',
    
    # Other
    'RP': 'O', 'POS': 'O', 'CD': 'O', 'UH': 'O', 'FW': 'O', 'SYM': 'O', '$': 'O', '#': 'O',
    
    # Punctuation
    '.': 'U', ',': 'U', ':': 'U'
}

# Convert to DataFrame
def words_with_pos(df, token_col='text'):
    """
    Takes a DataFrame with tokenized text and returns a DataFrame with:
    word | fine_pos | high_level_pos
    """
    rows = []
    for tokens in df[token_col]:
        pos_tags = nltk.pos_tag(tokens)
        for word, fine_pos in pos_tags:
            high_pos = pos_mapping.get(fine_pos, 'O')  # default 'O' if not mapped
            rows.append({'word': word, 'fine_pos': fine_pos, 'high_level_pos': high_pos})
    return pd.DataFrame(rows)

df_train_words = words_with_pos(df_train, token_col='text')
data_words = words_with_pos(data, token_col='text')

from nltk import pos_tag

nltk.download('averaged_perceptron_tagger_eng')

def get_high_level_pos(tag):
    if tag.startswith('N'):
        return 'N'
    elif tag.startswith('V'):
        return 'V'
    elif tag.startswith('J') or tag.startswith('R'):
        return 'A'
    elif tag in ['IN','TO','DT','PDT','CC','MD']:
        return 'P'
    elif tag in ['RP','POS','CD','UH','FW','SYM','$','#']:
        return 'O'
    elif tag in ['.',';',',',':']:
        return 'U'
    else:
        return 'O'

# Tag tokens in context and create new column
def tag_tokens(tokens):
    tagged = pos_tag(tokens)  # [('word','POS'), ...]
    high_level = [get_high_level_pos(tag) for _, tag in tagged]
    return high_level

data['pos_seq'] = data['text'].apply(tag_tokens)
df_train['pos_seq'] = df_train['text'].apply(tag_tokens)


def pos_counts(tokens):
    counts = {'N':0, 'V':0, 'A':0, 'P':0, 'O':0, 'U':0}
    for word, fine_pos in nltk.pos_tag(tokens):
        high_pos = pos_mapping.get(fine_pos, 'O') 
        counts[high_pos] += 1
    return counts

data['text'] = data['text'].apply(lambda x: x.split() if isinstance(x, str) else x)
df_train['text'] = df_train['text'].apply(lambda x: x.split() if isinstance(x, str) else x)

pos_features_data = data['text'].apply(pos_counts).apply(pd.Series)
pos_features_df_train = df_train['text'].apply(pos_counts).apply(pd.Series)
data = pd.concat([data, pos_features_data], axis=1)
df_train = pd.concat([df_train, pos_features_df_train], axis=1)


import ast
import pandas as pd

POS_TAGS = ["N", "V", "A", "P", "O", "U"]

# Create full bigram lexicon
BIGRAMS = [a + b for a in POS_TAGS for b in POS_TAGS]

def add_pos_bigrams(df):
    # Initialize all bigram columns
    for bg in BIGRAMS:
        df[bg] = 0

    # Fill counts
    for i, seq in df["pos_seq"].items():
        # Create bigrams from sequence
        seq_bigrams = [seq[j] + seq[j+1] for j in range(len(seq)-1)]

        # Count them
        counts = pd.Series(seq_bigrams).value_counts()

        # Assign to row
        for bg in BIGRAMS:
            df.at[i, bg] = counts.get(bg, 0)
    return df

data = add_pos_bigrams(data)
df_train = add_pos_bigrams(df_train)

data2=data.copy()


import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch

class model2_small_more_nodes(nn.Module):
    def __init__(self, vocab_size, pos_vocab_size, 
                 lstm_hidden=32, 
                 tfidf_dim=100, 
                 punct_dim=13,
                 use_tfidf=True, 
                 use_sentiment=True, 
                 use_punct=True, 
                 num_hidden=64,
                 output_size=2):
        super().__init__()

        self.use_tfidf = use_tfidf
        self.use_sentiment = use_sentiment
        self.use_punct = use_punct

        self.word_emb = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.pos_emb  = nn.Embedding(pos_vocab_size, 32, padding_idx=0)

        self.lstm = nn.LSTM(128+32, lstm_hidden, batch_first=True, bidirectional=True)

        aux_size = 0
        if use_tfidf:
            aux_size += tfidf_dim
        if use_sentiment:
            aux_size += 1
        if use_punct:
            aux_size += punct_dim

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden*2 + aux_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, word_ids, pos_ids, aux_features=None):
        w = self.word_emb(word_ids)
        p = self.pos_emb(pos_ids)
        x = torch.cat([w, p], dim=-1)

        out, _ = self.lstm(x)
        h = out.size(2)//2
        lstm_feat = torch.cat([out[:, -1, :h], out[:, 0, h:]], dim=1)

        if aux_features is not None:
            combined = torch.cat([lstm_feat, aux_features], dim=1)
        else:
            combined = lstm_feat

        return self.fc(combined)




def load_checkpoint(
    path,
    model_cls,
    optimizer_cls=None,
    scheduler_cls=None,
    device="cpu"
):
    checkpoint = torch.load(path, map_location=device)

    config = checkpoint["config"]
    model = model_cls(**config["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    optimizer = None
    if optimizer_cls is not None:
        optimizer = optimizer_cls(model.parameters(), lr=config["lr"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    scheduler = None
    if scheduler_cls is not None and checkpoint["scheduler_state"] is not None:
        scheduler = scheduler_cls(optimizer)
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "word_vocab": checkpoint["word_vocab"],
        "pos_vocab": checkpoint["pos_vocab"],
        "config": config
    }
    
checkpoint = load_checkpoint(
    "bilstm_model.pt",
    model_cls=model2_small_more_nodes,
    optimizer_cls=torch.optim.SGD, 
    scheduler_cls=lambda optim: torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1),
    device="cpu"
)

bilstm_model = checkpoint["model"]
optimizer = checkpoint["optimizer"]
scheduler = checkpoint["scheduler"]
word_vocab = checkpoint["word_vocab"]
pos_vocab  = checkpoint["pos_vocab"]

bilstm_model.eval()



# import pandas as pd
# df_train = pd.read_csv("train_features.csv")
# df_valid = pd.read_csv("valid_features.csv")
# df_test = pd.read_csv("test_features.csv")


import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import ast

def bilstm_processing(df_train, other_dfs, tfidf_data, svd_components=100):
    """
    Preprocess LSTM and TF-IDF data for stacking.
    
    Parameters:
    df_train : must contain features
    other_dfs : must contain features
        List of additional datasets (validation/test) to preprocess.
    tfidf_data : Paths to npz TF-IDF files corresponding to df_train + other_dfs.
 
    """
    assert len(other_dfs) + 1 == len(tfidf_data), "Must provide tfidf files for all datasets including train"
    
    # --- Load and transform TF-IDF ---
    # def load_npz(path):
    #     npz = np.load(path)
    #     return csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=tuple(npz["shape"]))
    
    def load_npz(path_or_matrix):
        if isinstance(path_or_matrix, csr_matrix):
            return path_or_matrix
        
        npz = np.load(path_or_matrix)
        return csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=tuple(npz["shape"]))

    
    X_tfidf_train = load_npz(tfidf_data[0])
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_svd_train = svd.fit_transform(X_tfidf_train)

    
    
    # --- Process text columns ---
    # for df in [df_train] + other_dfs:
        # df["text"] = df["text"].apply(ast.literal_eval)
        # df["pos_seq"] = df["pos_seq"].apply(ast.literal_eval)
    
    # --- Build vocabularies from training data ---
    def build_vocab(tokens_list, min_freq=1):
        counter = Counter()
        for sentence in tokens_list:
            for token in sentence:
                counter.update([token])
        vocab = {"<pad>": 0, "<unk>": 1}
        for tok, freq in counter.items():
            if freq >= min_freq:
                vocab[tok] = len(vocab)
        return vocab
    
    word_vocab = build_vocab(df_train["text"])
    pos_vocab = build_vocab(df_train["pos_seq"])
    
    # --- Encode sequences ---
    def encode(tokens, vocab):
        return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    
    for df in [df_train] + other_dfs:
        df["word_ids"] = df["text"].apply(lambda x: encode(x, word_vocab))
        df["pos_ids"]  = df["pos_seq"].apply(lambda x: encode(x, pos_vocab))
    
    # --- Pad sequences ---
    max_len = max(len(seq) for seq in df_train["text"])
    
    def pad(seq, length, pad_value=0):
        return seq + [pad_value] * (length - len(seq))
    
    for df in [df_train] + other_dfs:
        df["word_padded"] = df["word_ids"].apply(lambda x: pad(x, max_len))
        df["pos_padded"]  = df["pos_ids"].apply(lambda x: pad(x, max_len))
    
    # --- TF-IDF SVD features ---
    X_svd_others = []
    for i, df in enumerate(other_dfs, start=1):
        X_tfidf_other = load_npz(tfidf_data[i])
        X_svd_other = svd.transform(X_tfidf_other)
        X_svd_others.append(X_svd_other)
    
    # --- Assign SVD features to DataFrames ---
    df_train["tfidf_svd"] = list(X_svd_train)
    df_train["tfidf_order"] = [X_svd_train[i] for i in range(X_svd_train.shape[0])]

    for df, X_svd_other in zip(other_dfs, X_svd_others):
        df["tfidf_svd"] = list(X_svd_other)
        df["tfidf_order"] = [X_svd_other[i] for i in range(X_svd_other.shape[0])]

        
    
    # --- Output dictionary ---
    processed = {
        "word_vocab": word_vocab,
        "pos_vocab": pos_vocab,
        "train": df_train,
        #     {
        #     "word_padded": np.stack(df_train["word_padded"].values),
        #     "pos_padded": np.stack(df_train["pos_padded"].values),
        #     "tfidf_svd": np.stack(df_train["tfidf_svd"].values),
        #     "labels": df_train["label"].values
        # },
        "others": other_dfs,

            #[
        #     {
        #         "word_padded": np.stack(df["word_padded"].values),
        #         "pos_padded": np.stack(df["pos_padded"].values),
        #         "tfidf_svd": np.stack(df["tfidf_svd"].values),
        #         "labels": df["label"].values
        #     }
        #     for df in other_dfs
        # ],
        "svd": svd
    }
    
    return processed

processed = bilstm_processing(
    df_train=df_train,
    other_dfs=[data],
    tfidf_data=[train_tfidf, data_tfidf]
)

train_data = processed["train"]
data_processed = processed["others"][0]
# test_data  = processed["others"][1]

# word_vocab = processed['word_vocab']
# pos_vocab = processed['pos_vocab']


from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

from torch.utils.data import Dataset
import torch

class TextPosDataset(Dataset):
    def __init__(self, df, use_tfidf=True, use_sentiment=True, use_punct=True):
        self.word_ids = df["word_ids"].tolist()
        self.pos_ids = df["pos_ids"].tolist()
        self.label = df["label"].tolist()

        self.use_tfidf = use_tfidf
        self.use_sentiment = use_sentiment
        self.use_punct = use_punct

        if use_tfidf:
            self.tfidf = df["tfidf_order"].tolist()
        if use_sentiment:
            self.sentiment = df["sentiment"].tolist()
        if use_punct:
            self.punct_features = df[[
                'exclamation_count','question_count','ellipsis_count','quote_count',
                'comma_count','period_count','semicolon_count','colon_count','dash_count',
                'multiple_exclamation','multiple_question','mixed_punctuation','punct_density'
            ]].values.tolist()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        items = [
            torch.tensor(self.word_ids[idx]),
            torch.tensor(self.pos_ids[idx])
        ]

        # optional auxiliary features
        aux_list = []
        if self.use_tfidf:
            aux_list.append(torch.tensor(self.tfidf[idx], dtype=torch.float))
        if self.use_sentiment:
            aux_list.append(torch.tensor([self.sentiment[idx]], dtype=torch.float))
        if self.use_punct:
            aux_list.append(torch.tensor(self.punct_features[idx], dtype=torch.float))

        # concatenate all auxiliary features
        if aux_list:
            aux_features = torch.cat(aux_list)
        else:
            aux_features = torch.tensor([])  # empty tensor if none selected

        items.append(aux_features)
        items.append(torch.tensor(self.label[idx]))
        return tuple(items)

def collate(batch):
    words, pos, aux_features, labels = zip(*batch)
    
    # pad sequences
    words = rnn_utils.pad_sequence(words, batch_first=True, padding_value=0)
    pos   = rnn_utils.pad_sequence(pos, batch_first=True, padding_value=0)
    
    if aux_features[0].numel() > 0:
        aux_features = torch.stack(aux_features)
    else:
        aux_features = None  # no aux features used

    labels = torch.stack(labels)
    return words, pos, aux_features, labels


valid_dataset = TextPosDataset(data)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate)

# test_dataset = TextPosDataset(df_test)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)



import torch

def get_lstm_predictions(model, data_loader, device="cpu"):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for words, pos, aux, labels in data_loader:
            words, pos, aux = words.long().to(device), pos.long().to(device), aux.float().to(device)
            outputs = model(words, pos, aux)  # shape [batch_size, num_classes]
            probs = torch.softmax(outputs, dim=1)  # convert logits to probabilities
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.vstack(all_probs)  # shape [num_samples, num_classes]
    all_labels = np.concatenate(all_labels)  # shape [num_samples]
    return all_probs, all_labels

val_probs, val_labels = get_lstm_predictions(bilstm_model, valid_loader, device="cpu")

validation_predictions = pd.DataFrame({
    "prediction": np.argmax(val_probs, axis=1),
    "probability": np.max(val_probs, axis=1),
    "label": val_labels
})
validation_predictions.to_csv("bilstm_Newdata_predictions.csv", index=False)



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

print("Validation Accuracy:", accuracy_score(val_labels, np.argmax(val_probs, axis=1)))
print("Validation Confusion Matrix:\n", confusion_matrix(val_labels, np.argmax(val_probs, axis=1)))
print("Validation Classification Report:\n", classification_report(val_labels, np.argmax(val_probs, axis=1)))


from scipy.sparse import load_npz, hstack, csr_matrix
from joblib import load


mlp = load("final_mlp.pkl")
mlp_scaler = load("mlp_scaler.pkl")

numeric_cols = data2.columns.difference(
    ['text','tokens','pos_seq','label']
)

X_num = data2[numeric_cols].astype(float).values
X_num = mlp_scaler.transform(X_num)

X_test = hstack([
    data_tfidf,
    csr_matrix(X_num)
]).toarray()

pred = mlp.predict(X_test)
proba = mlp.predict_proba(X_test)

confidence = proba[np.arange(len(pred)), pred]

mlp_newdata_predictions=pd.DataFrame({
    "prediction": pred,
    "probability": confidence,
    "label": data2["label"].values
})

mlp_newdata_predictions.to_csv("mlp_newdata_predictions.csv", index=False)

nb_model = load("final_NB.pkl") 


pred = nb_model.predict(data_tfidf)
proba = nb_model.predict_proba(data_tfidf)
confidence = proba[np.arange(len(pred)), pred]

# Save predictions
nb_newdata_predictions = pd.DataFrame({
    "prediction": pred,
    "probability": confidence,
    "label": data2["label"].values
})

nb_newdata_predictions.to_csv("naiveBayes_newdata_predictions.csv", index=False)


svd = load("svd_RF.joblib")
rf_model = load("final_RF.pkl")

# Transform TF-IDF for new data
X_svd = svd.transform(data_tfidf)

# Numeric features (reuse your existing numeric_cols)
X_num = data2[numeric_cols].astype(float).values

# Combine SVD + numeric features
X_test = np.hstack([X_svd, X_num])

# Predict
pred = rf_model.predict(X_test)
proba = rf_model.predict_proba(X_test)
confidence = proba[np.arange(len(pred)), pred]

# Save predictions
rf_newdata_predictions = pd.DataFrame({
    "prediction": pred,
    "probability": confidence,
    "label": data2["label"].values
})
rf_newdata_predictions.to_csv("randomForest_newdata_predictions.csv", index=False)



svd = load("svd_LR.joblib")
scaler = load("scaler_LR")
lr_model = load("final_LR.pkl")

X_num = data2[numeric_cols].astype(float).values

# Apply SVD to TF-IDF
X_svd = svd.transform(data_tfidf)

# Combine SVD features with numeric features
X_combined = np.hstack([X_svd, X_num])

# Scale features
X_scaled = scaler.transform(X_combined)

# Predict
preds = lr_model.predict(X_scaled)
probs = lr_model.predict_proba(X_scaled)
pred_class_probs = probs[np.arange(len(preds)), preds]

# Save predictions
lr_newdata_predictions = pd.DataFrame({
    "prediction": preds,
    "probability": pred_class_probs,
    "label": data2["label"].values  # optional if labels exist
})

lr_newdata_predictions.to_csv("logisticRegression_newdata_predictions.csv", index=False)

# ======================
# CNN/MLP Predictions
# ======================
cnn_model = load("final_cnn.pkl")
svd_cnn = load("svd_cnn.joblib")
scaler_cnn = load("scaler_cnn.joblib")

# Get numeric features
X_num = data2[numeric_cols].astype(np.float64).values

# Apply SVD to TF-IDF
X_svd = svd_cnn.transform(data_tfidf)

# Combine SVD features with numeric features
X_combined = np.hstack([X_svd, X_num])

# Scale features
X_scaled = scaler_cnn.transform(X_combined)

# Predict
preds = cnn_model.predict(X_scaled)
probs = cnn_model.predict_proba(X_scaled)
pred_class_probs = probs[np.arange(len(preds)), preds]

# Save predictions
cnn_newdata_predictions = pd.DataFrame({
    "prediction": preds,
    "probability": pred_class_probs,
    "label": data2["label"].values
})
cnn_newdata_predictions.to_csv("cnn_newdata_predictions.csv", index=False)
print("✓ CNN/MLP predictions saved!")

# ======================
# SVM Predictions
# ======================
from joblib import load

svm_model = load("final_svm.pkl")
svd_svm = load("svd_svm.joblib")
scaler_svm = load("scaler_svm.joblib")

# Get numeric features
numeric_cols = data2.columns.difference(['text','tokens','pos_seq','label'])
X_num = data2[numeric_cols].astype(np.float64).values

# Apply SVD to TF-IDF
X_svd = svd_svm.transform(data_tfidf)

# Combine SVD features with numeric features
X_combined = np.hstack([X_svd, X_num])

# Scale features
X_scaled = scaler_svm.transform(X_combined)

# Predict
preds = svm_model.predict(X_scaled)
probs = svm_model.predict_proba(X_scaled)
pred_class_probs = probs[np.arange(len(preds)), preds]

# Save predictions
svm_newdata_predictions = pd.DataFrame({
    "prediction": preds,
    "probability": pred_class_probs,
    "label": data2["label"].values
})
svm_newdata_predictions.to_csv("svm_newdata_predictions.csv", index=False)
print("✓ SVM predictions saved!")