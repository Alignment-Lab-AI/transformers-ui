import torch
from transformers import *
from data import * 
from models import *

def train_model(dataset, model_name, n_epochs=3):
    """Train a model on a dataset"""
    # Load model, tokenizer and dataset
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                              num_labels=len(dataset[2].features['labels'].unique()))
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    train_dataset, val_dataset, test_dataset = dataset
    
    # Define optimizer, criterion and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                               num_warmup_steps=0,  
                                               num_training_steps=n_epochs*len(train_dataset))  
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Train model
    training_progress = {}
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataset: 
            # Forward pass
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        # Evaluation
        model.eval()
        val_loss, val_acc = eval_model(val_dataset, model, criterion)
        print(f'Epoch {epoch+1} - Loss: {train_loss/len(train_dataset)} Accuracy: {val_acc*100:.2f}%')   
        
        # Save metrics
        training_progress[epoch] = {
            'train_loss': train_loss/len(train_dataset), 
            'val_loss': val_loss/len(val_dataset),
            'val_acc': val_acc 
        }
        
    # Save model and metrics 
    save_model(model, model_name)
    save_metrics(training_progress, model_name)

def save_model(model, model_name):
    """Save trained model to cache"""
    MODEL_CACHE[model_name] = model

def save_metrics(metrics, model_name):
    """Save training metrics to cache"""
    METRICS_CACHE[model_name] = metrics 
    
def eval_model(dataset, model, criterion):  
    """Evaluate a model on a validation dataset"""
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in dataset:  
            # Forward pass
            outputs = model(**batch) 
            # Calculate loss
            loss = criterion(outputs.logits, batch['labels'])
            val_loss += loss.item()
            # Get predictions 
            predictions = torch.argmax(outputs.logits, dim=-1)
            # Calculate accuracy
            correct += torch.sum(predictions == batch['labels'])  
    
    val_acc = correct.double() / len(dataset)
    return val_loss, val_acc 
