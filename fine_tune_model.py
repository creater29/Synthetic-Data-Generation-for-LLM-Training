import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load pre-trained model
def load_model(model_name="distilbert-base-uncased"):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare the dataset
def prepare_dataset(data):
    dataset = load_dataset('csv', data_files={'train': data}, delimiter=',')
    train_dataset, eval_dataset = train_test_split(dataset['train'], test_size=0.2)
    return train_dataset, eval_dataset

# Fine-tuning the model
def fine_tune_model(train_dataset, eval_dataset, model):
    training_args = TrainingArguments(
        output_dir='./results', 
        num_train_epochs=3, 
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8, 
        warmup_steps=500, 
        weight_decay=0.01, 
        logging_dir='./logs', 
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset
    )
    
    trainer.train()

# Main function
if __name__ == "__main__":
    # Load synthetic data
    train_data = 'data/synthetic_data.csv'
    
    # Load model
    model = load_model()

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(train_data)

    # Fine-tune model
    fine_tune_model(train_dataset, eval_dataset, model)

    # Save fine-tuned model
    model.save_pretrained('models/fine_tuned_model')
