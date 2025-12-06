import pandas as pd


def join_tokens(token_list_str):
    # Convert "['a','b','c']" → ['a','b','c']
    tokens = eval(token_list_str)
    return " ".join(tokens)


train_df = pd.read_csv("train_features.csv") 
valid_df = pd.read_csv("valid_features.csv") 
test_df  = pd.read_csv("test_features.csv")  

train_df["text_joined"] = train_df.iloc[:,0].apply(join_tokens)
valid_df["text_joined"] = valid_df.iloc[:,0].apply(join_tokens)
test_df["text_joined"]  = test_df.iloc[:,0].apply(join_tokens)
    
y_train = train_df.iloc[:,1]
y_valid = valid_df.iloc[:,1]

print (train_df.head())
print(y_train.head())