import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, learning_curve  
import plotly.express as px
import pandas as pd

def compute_metrics(model, dataset):
    """Compute metrics on predictions made by a model on a dataset"""
    model.eval()
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
    return {
        'val_loss': val_loss/len(dataset),
        'val_acc': val_acc 
    }

def plot_metrics(metrics, model_name):
    """Plot training progress metrics"""
    metrics_df = pd.DataFrame(metrics) 
    fig = px.line(metrics_df, x=metrics_df.index, y=['val_loss', 'val_acc'])
    fig.update_layout(
        title=f'Training Metrics for {model_name}', 
        xaxis_title='Epoch',
        yaxis_title='Metric Value'
    )
    st.plotly_chart(fig)  
    
def show_evaluation(model, dataset):
    """Show model evaluation results on a dataset"""
    metrics = compute_metrics(model, dataset)
    st.subheader("Evaluation results")
    st.table(pd.DataFrame(metrics, index=[0])) 
    plot_metrics(metrics, model.model_name)

def plot_confusion_matrix(model, dataset, target_names):
    """Plot a confusion matrix""" 
    y_true = [d['labels'] for d in dataset]
    predictions = torch.argmax(model.outputs, dim=-1)
    cm = confusion_matrix(y_true, predictions)
    cm_df = pd.DataFrame(cm, 
                         index=target_names, 
                         columns=target_names)
    fig = px.imshow(cm_df, 
                    title='Confusion Matrix',
                    labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(fig)

def show_classification_report(model, dataset, target_names):
    """Show precision, recall and F1 score"""
    y_true = [d['labels'] for d in dataset]
    y_pred = model.predictions  
    report = classification_report(y_true, y_pred, target_names=target_names)
    st.table(report)

def plot_precision_recall_curve(model, dataset):
    """Plot the precision-recall curve"""
    y_true = [d['labels'] for d in dataset]
    y_scores = model.predictions 
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    fig = px.line(x=thresholds, y=precisions, title='Precision-Recall Curve')
    st.plotly_chart(fig)

def plot_roc_curve(model, dataset):
    """Plot the ROC curve"""
    y_true = [d['labels'] for d in dataset]
    y_scores = model.predictions 
    false_positive_rates, true_positive_rates, thresholds = roc_curve(y_true, y_scores)  
    fig = px.line(x=false_positive_rates, y=true_positive_rates, title='ROC Curve')
    st.plotly_chart(fig) 
    
def plot_learning_curve(model, dataset):  
    """Plot learning curves"""
    train_sizes, train_scores, valid_scores = learning_curve(model, 
                                                            dataset, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            random_state=42)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    fig = px.line(x=train_sizes, 
                  y=train_mean, 
                  title='Learning Curves',
                  labels={'train_sizes':'Training examples (% of dataset)', 
                            'Score':'Performance score'})
    fig.add_scatter(x=train_sizes, y=valid_mean, mode='lines', name='Validation')
    fig.add_vrect(x0=0, x1=max(train_sizes), 
                  y0=train_mean[0]-train_std[0], y1=train_mean[0]+train_std[0], 
                  fillcolor="green", opacity=0.25, line_width=0)
    fig.add_vrect(x0=0, x1=max(train_sizes), 
                  y0=valid_mean[0]-valid_std[0], y1=valid_mean[0]+valid_std[0], 
                  fillcolor="red", opacity=0.25, line_width=0)
    st.plotly_chart(fig)  

def show_advanced_metrics(model, dataset, target_names):
    """Show advanced metrics - confusion matrix, classification report, PR and ROC curves, learning curves"""
    plot_confusion_matrix(model, dataset, target_names) 
    show_classification_report(model, dataset, target_names)
    plot_precision_recall_curve(model, dataset)
    plot_roc_curve(model, dataset)
    plot_learning_curve(model, dataset) 
