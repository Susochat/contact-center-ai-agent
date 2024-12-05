from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Trainer,
    TrainingArguments
)
import torch
from datasets import load_dataset

# Define human-readable broader sentiment categories
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

# Load models and tokenizers
def load_models():
    sentiment_model_name = "distilbert-base-uncased"
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, num_labels=3)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

    return {
        "sentiment_model": sentiment_model,
        "sentiment_tokenizer": sentiment_tokenizer,
    }

models = load_models()
print("Models loaded successfully!")

def fine_tune_sentiment_model(train_dataset, eval_dataset):
    """Fine-tune the sentiment analysis model."""
    training_args = TrainingArguments(
        output_dir="./sentiment_model_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        logging_dir="./logs",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=models["sentiment_model"],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

def load_and_process_data():
    """Load and process data for fine-tuning the sentiment model."""
    dataset_path = "../data/contact_center_ai_agent_dataset.json"
    
    # Assuming your dataset is in a JSON format, load it into a Hugging Face Dataset
    dataset = load_dataset("json", data_files=dataset_path, split='train')

    # Define the sentiment mapping (from string to integer)
    SENTIMENT_LABELS_MAP = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2,
    }

    # Tokenize the dataset
    def tokenize_function(examples):
        return models["sentiment_tokenizer"](examples["text"], padding="max_length", truncation=True, max_length=128)

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Map the 'sentiment' to integer labels
    tokenized_datasets = tokenized_datasets.map(lambda x: {"label": SENTIMENT_LABELS_MAP.get(x["sentiment"], -1)}, batched=True)

    # Ensure the dataset has input_ids and attention_mask
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Split the dataset into train and eval
    dataset_split = tokenized_datasets.train_test_split(test_size=0.1)

    return dataset_split['train'], dataset_split['test']

# Load and process data
train_dataset, eval_dataset = load_and_process_data()
print(f"Train dataset: {train_dataset}")
print(f"Eval dataset: {eval_dataset}")

# Fine-tune the sentiment model
fine_tune_sentiment_model(train_dataset, eval_dataset)

def analyze_sentiment(text):
    """Perform sentiment analysis using the fine-tuned model."""
    sentiment_pipeline = pipeline("sentiment-analysis", model=models["sentiment_model"], tokenizer=models["sentiment_tokenizer"])
    result = sentiment_pipeline(text)
    
    sentiment_score = result[0]["label"]
    sentiment_index = int(sentiment_score.split("_")[-1])  # Extracts number from 'LABEL_0', 'LABEL_1', etc.
    sentiment_category = SENTIMENT_LABELS.get(sentiment_index, "Unknown")

    return sentiment_category, result[0]["score"]

# Flask app setup
app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze text for sentiment using the fine-tuned model."""
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    sentiment_category, sentiment_score = analyze_sentiment(text)

    response = {
        "text": text,
        "sentiment": {
            "label": sentiment_category,
            "score": sentiment_score,
        },
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=True)
from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Trainer,
    TrainingArguments
)
import torch
from datasets import load_dataset

# Define human-readable broader sentiment categories
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

# Load models and tokenizers
def load_models():
    sentiment_model_name = "distilbert-base-uncased"
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, num_labels=3)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

    return {
        "sentiment_model": sentiment_model,
        "sentiment_tokenizer": sentiment_tokenizer,
    }

models = load_models()
print("Models loaded successfully!")

def fine_tune_sentiment_model(train_dataset, eval_dataset):
    """Fine-tune the sentiment analysis model."""
    training_args = TrainingArguments(
        output_dir="./sentiment_model_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=models["sentiment_model"],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

def load_and_process_data():
    """Load and process data for fine-tuning the sentiment model."""
    dataset_path = "../data/contact_center_ai_agent_dataset.json"
    
    # Assuming your dataset is in a JSON format, load it into a Hugging Face Dataset
    dataset = load_dataset("json", data_files=dataset_path, split='train')

    # Define the sentiment mapping (from string to integer)
    SENTIMENT_LABELS_MAP = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2,
    }

    # Tokenize the dataset
    def tokenize_function(examples):
        return models["sentiment_tokenizer"](examples["text"], padding="max_length", truncation=True, max_length=128)

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Handle 'sentiment' as a list or single value
    def process_sentiment(x):
        sentiment_value = x["sentiment"]
        
        # Check if sentiment is a list, and extract the first element if it is
        if isinstance(sentiment_value, list):
            sentiment_value = sentiment_value[0]  # Modify this logic if the list has other elements to consider
        
        # Map sentiment value to integer label
        return {"label": SENTIMENT_LABELS_MAP.get(sentiment_value, -1)}

    # Apply sentiment mapping
    tokenized_datasets = tokenized_datasets.map(process_sentiment, batched=True)

    # Ensure the dataset has input_ids and attention_mask
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Split the dataset into train and eval
    dataset_split = tokenized_datasets.train_test_split(test_size=0.1)

    return dataset_split['train'], dataset_split['test']



# Load and process data
train_dataset, eval_dataset = load_and_process_data()
print(f"Train dataset: {train_dataset}")
print(f"Eval dataset: {eval_dataset}")

# Fine-tune the sentiment model
fine_tune_sentiment_model(train_dataset, eval_dataset)

def analyze_sentiment(text):
    """Perform sentiment analysis using the fine-tuned model."""
    sentiment_pipeline = pipeline("sentiment-analysis", model=models["sentiment_model"], tokenizer=models["sentiment_tokenizer"])
    result = sentiment_pipeline(text)
    
    sentiment_score = result[0]["label"]
    sentiment_index = int(sentiment_score.split("_")[-1])  # Extracts number from 'LABEL_0', 'LABEL_1', etc.
    sentiment_category = SENTIMENT_LABELS.get(sentiment_index, "Unknown")

    return sentiment_category, result[0]["score"]

# Flask app setup
app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze text for sentiment using the fine-tuned model."""
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    sentiment_category, sentiment_score = analyze_sentiment(text)

    response = {
        "text": text,
        "sentiment": {
            "label": sentiment_category,
            "score": sentiment_score,
        },
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=True)
