import torch 
from transformers import *

# Cache to store models    
MODEL_CACHE = {}  

def save_model_file(file):
    """Save uploaded model file to cache"""
    model_name = file.name 
    MODEL_CACHE[model_name] = file

def get_model_details(model_name):
    """Load model details from cache"""
    model = torch.load(MODEL_CACHE[model_name])
    details = {
        'Model name': model_name,
        'Model architecture': model['arch'], 
        'Loss function': model['criterion'],
        'Optimizer': model['optimizer_name'],
        'Learning rate': model['lr'],
        'Training epochs': model['n_epochs']
    }  
    return pd.DataFrame(details, index=[0])

def list_models():
    """Return list of model names in cache"""
    return list(MODEL_CACHE.keys())   

def clear_cache():
    """Clear cached models"""
    MODEL_CACHE.clear()

def view_model_details(model_name):
    """View details of a cached model"""
    model = torch.load(MODEL_CACHE[model_name])
    details = get_model_details(model_name)
    st.dataframe(details)
    st.write(f"Sample predictions: {view_model_predictions(model, inputs)}")

def view_model_predictions(model, inputs):
    """View predictions made by a model on input data"""
    # Encode inputs
    inputs = tokenizer.batch_encode_plus(inputs, 
                                         add_special_tokens=True,
                                         return_tensors='pt')     
    
    # Get predictions 
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Decode predictions
    prediction = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    return prediction 
    
def train_model(dataset, model_name, n_epochs=3):
    """Train a model on a dataset"""
    # Load model, tokenizer and dataset
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                              num_labels=len(dataset[2]))
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    train_dataset, val_dataset, _ = dataset
    
    # Trainer, scheduler and optimizer
    trainer = Trainer(model, 
                      AdamW(model.parameters(), lr=1e-3),
                      n_epochs=n_epochs)
    
    # Train model
    trainer.train(train_dataset, val_dataset)  
    return trainer.model  

def inference(inputs, model):
    """Make predictions using a trained model"""
    # Encode inputs
    inputs = tokenizer.batch_encode_plus(inputs, 
                                         add_special_tokens=True,
                                         return_tensors='pt')   
    
    # Get predictions
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Decode predictions
    prediction = tokenizer.batch_decode(predictions, skip_special_tokens=True) 
    return prediction
