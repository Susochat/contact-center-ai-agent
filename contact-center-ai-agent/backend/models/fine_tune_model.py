from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load the processed dataset
dataset = load_dataset('json', data_files={'train': '../data/contact_center_processed_dataset.json'}, split='train')

# Split dataset into train and validation sets using Hugging Face's dataset library
split_dataset = dataset.train_test_split(test_size=0.1)

# Extract the train and validation datasets
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')  # Use smaller model for memory efficiency

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    # Concatenate all conversation turns into a single string for each example
    conversation_texts = [
        " ".join([turn["text"] for turn in conv]) for conv in examples["conversation"]
    ]
    # Tokenize the concatenated conversation strings
    encoding = tokenizer(conversation_texts, padding="max_length", truncation=True, max_length=256)  # Reduced max_length
    # Set labels to be the same as input_ids for causal language modeling
    encoding['labels'] = encoding['input_ids']
    return encoding

# Apply the tokenization
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Load the pre-trained model (DistilGPT2)
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

# Define training arguments with smaller batch size and gradient accumulation
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    save_strategy="epoch",  
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduced batch size
    per_device_eval_batch_size=2,   # Reduced batch size
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    gradient_accumulation_steps=4,  # Simulate a larger batch size
    fp16=False,  # Disable mixed precision since MPS doesn't support it
)

# Set up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  # Pass the eval dataset here
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

print("Fine-tuning complete and model saved!")
