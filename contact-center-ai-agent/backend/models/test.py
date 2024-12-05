from flask import Flask, request, jsonify
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


# Define sentiment labels
SENTIMENT_LABELS_MAP = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}

# Load models and tokenizers (for sentiment)
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, num_labels=3, ignore_mismatched_sizes=True)

# Load and process data
def load_and_process_data():
    """Load and process data for fine-tuning the sentiment model."""
    dataset_path = "../data/contact_center_processed_dataset.json"
    
    # Load dataset from a JSON file
    dataset = load_dataset("json", data_files=dataset_path, split='train')

    # Preprocess and handle the 'sentiment' field properly
    def preprocess_sentiment(example):
        # If 'sentiment' is a list, get the first element; otherwise, leave it as is
        sentiment_value = example["sentiment"]
        if isinstance(sentiment_value, list):
            sentiment_value = sentiment_value[0]  # Extract the first sentiment value
        
        # Map the sentiment value to its corresponding label
        example["label"] = SENTIMENT_LABELS_MAP.get(sentiment_value, -1)
        return example

    # Apply the sentiment preprocessing to the dataset
    processed_dataset = dataset.map(preprocess_sentiment)

    # Tokenization
    def tokenize_function(examples):
        return sentiment_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    # Apply tokenization
    tokenized_datasets = processed_dataset.map(tokenize_function, batched=True)

    # Ensure the dataset is in the right format for PyTorch
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Split the dataset into train and eval
    dataset_split = tokenized_datasets.train_test_split(test_size=0.1)

    return dataset_split['train'], dataset_split['test']

# Fine-tune the sentiment model
def fine_tune_sentiment_model(train_dataset, eval_dataset):
    """Fine-tune the sentiment analysis model."""
    from transformers import Trainer, TrainingArguments

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
        model=sentiment_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

# Function to perform sentiment analysis
def analyze_sentiment(text):
    """Perform sentiment analysis using the fine-tuned model."""
    sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
    result = sentiment_pipeline(text)
    
    sentiment_score = result[0]["label"]
    sentiment_index = int(sentiment_score.split("_")[-1])  # Extracts number from 'LABEL_0', 'LABEL_1', etc.
    sentiment_category = list(SENTIMENT_LABELS_MAP.keys())[sentiment_index]  # Map to human-readable sentiment

    return sentiment_category, result[0]["score"]

# Flask app setup
app = Flask(__name__)

# Route to perform sentiment analysis
@app.route('/status', methods=['GET'])
def status():
    return "Route works!"
@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze text for sentiment using fine-tuned model."""
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

@app.route("/train", methods=["POST"])
def train_model():
    """Route to start the training of the sentiment model."""
    train_dataset, eval_dataset = load_and_process_data()

    fine_tune_sentiment_model(train_dataset, eval_dataset)

    return jsonify({"status": "Training complete!"})

# Main entry point
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=True)
