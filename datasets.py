import streamlit as st
import pandas as pd

# Cache to store datasets
DATASET_CACHE = {} 

def save_dataset_file(file):
    """Save uploaded dataset file to cache"""
    dataset_name = file.name 
    DATASET_CACHE[dataset_name] = file

def get_dataset_details(dataset_name):
    """Load dataset details from cache"""
    dataset_file = DATASET_CACHE[dataset_name]
    df = pd.read_csv(dataset_file)
    details = {
        'Dataset name': dataset_name,
        'Rows': df.shape[0],
        'Columns': df.shape[1],
        'Features': list(df.columns)
    }
    return pd.DataFrame(details, index=[0])

def list_datasets():
    """Return list of dataset names in cache"""
    return list(DATASET_CACHE.keys())  

def clear_cache():
    """Clear cached datasets"""
    DATASET_CACHE.clear()

def view_dataset(dataset_name): 
    """View the contents of a cached dataset"""
    dataset_file = DATASET_CACHE[dataset_name]
    df = pd.read_csv(dataset_file)
    st.dataframe(df)

def encode_dataset(dataset_file, model_name):
    """Encode a dataset for a given model"""
    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_dataset = tokenizer.batch_encode_plus(df, 
                                                 add_special_tokens=True,
                                                 return_attention_mask=True, 
                                                 pad_to_max_length=True, 
                                                 max_length=512,
                                                 return_tensors='pt')
    return encoded_dataset    

def tokenize_dataset(dataset_file):
    """Tokenize a dataset using a BERT tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenized_dataset = tokenizer.batch_encode_plus(df,  
                                                   add_special_tokens=True,
                                                   max_length=512, 
                                                   return_attention_mask=True,
                                                   return_tensors='pt')
    return tokenized_dataset  

def create_train_val_test_split(encoded_dataset):
    """Split encoded dataset into train, val and test splits"""
    # Split into splits
    train_size = int(0.8 * len(encoded_dataset))
    val_size = int(0.1 * len(encoded_dataset))
    test_size = len(encoded_dataset) - train_size - val_size   
    train_dataset = encoded_dataset[:train_size]
    val_dataset = encoded_dataset[train_size:train_size+val_size]
    test_dataset = encoded_dataset[-test_size:]
    return train_dataset, val_dataset, test_dataset 
