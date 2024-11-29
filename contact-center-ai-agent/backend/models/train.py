from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as the padding token

# Function to load a custom dataset
def load_custom_dataset():
    dataset = load_dataset(
        'json', 
        data_files={'train': '../data/contact_center_synthetic_dataset.json', 'test': '../data/test.json'}
    )
    return dataset

# Tokenize the dataset
def tokenize_function(examples):
    # If the dataset contains a nested structure like 'turns', flatten it
    texts = [turn['text'] for conversation in examples['turns'] for turn in conversation]
    # Tokenize the text and return the result
    return tokenizer(texts, truncation=True, padding="longest", max_length=512)

# Training the model
def train_model():
    # Load dataset
    dataset = load_custom_dataset()

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory for model and logs
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        learning_rate=5e-5,  # Learning rate for the optimizer
        per_device_train_batch_size=2,  # Batch size per GPU/CPU for training
        per_device_eval_batch_size=2,  # Batch size per GPU/CPU for evaluation
        num_train_epochs=3,  # Number of training epochs
        weight_decay=0.01,  # Regularization parameter
        logging_dir='./logs',  # Directory for logs
        logging_steps=10,  # Log every 10 steps
        save_steps=100,  # Save model every 100 steps
        save_total_limit=2,  # Limit the number of saved models
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="loss",  # Metric to monitor for the best model
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,  # Model to train
        args=training_args,  # Training arguments
        train_dataset=tokenized_dataset['train'],  # Training dataset
        eval_dataset=tokenized_dataset['test'],  # Evaluation dataset
    )

    # Start the training process
    trainer.train()

if __name__ == "__main__":
    train_model()
