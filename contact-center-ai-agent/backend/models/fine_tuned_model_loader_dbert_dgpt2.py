from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline,
    Trainer,
    TrainingArguments
)
import torch
from datasets import load_dataset

# Define human-readable broader intent labels and sentiment categories
INTENT_LABELS = {
    0: "Order Management",  # Includes: Order Status, Replacement Request, Return Process, Order Modification
    1: "Technical Support",  # Includes: Technical Support, Assistance, Service Cancellation
    2: "Account & Billing",  # Includes: Account Access, Billing Inquiry, Payment Issues, Account Update
    3: "Product Inquiry",  # Includes: Product Availability, Promotions, Warranty Information
    4: "General Inquiry & Feedback",  # Includes: Shipping Inquiry, Feedback, Complaint, General Inquiry
}

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

    intent_model_name = "distilbert-base-uncased"  # Placeholder for intent recognition fine-tuned model
    intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name, num_labels=5)
    intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)

    response_model_name = "distilgpt2"
    response_model = AutoModelForCausalLM.from_pretrained(response_model_name)
    response_tokenizer = AutoTokenizer.from_pretrained(response_model_name)

    # Summarization model (Bart)
    summarization_model_name = "facebook/bart-large-cnn"
    summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name)
    summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name)

    return {
        "sentiment_model": sentiment_model,
        "sentiment_tokenizer": sentiment_tokenizer,
        "intent_model": intent_model,
        "intent_tokenizer": intent_tokenizer,
        "response_model": response_model,
        "response_tokenizer": response_tokenizer,
        "summarization_model": summarization_model,
        "summarization_tokenizer": summarization_tokenizer,
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

def fine_tune_intent_model(train_dataset, eval_dataset):
    """Fine-tune the intent recognition model."""
    training_args = TrainingArguments(
        output_dir="./intent_model_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=models["intent_model"],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

def fine_tune_response_model(train_dataset):
    """Fine-tune the response generation model."""
    training_args = TrainingArguments(
        output_dir="./response_model_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=models["response_model"],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Evaluation can be done with the same dataset for simplicity
        tokenizer=models["response_tokenizer"],
    )

    trainer.train()

def fine_tune_summarization_model(train_dataset):
    """Fine-tune the summarization model (Bart)."""
    training_args = TrainingArguments(
        output_dir="./summarization_model_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=models["summarization_model"],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Evaluation can be done with the same dataset for simplicity
        tokenizer=models["summarization_tokenizer"],
    )

    trainer.train()

# Load and process data
def load_and_process_data():
    # Load dataset from JSON file
    dataset_path = "../data/contact_center_ai_agent_dataset.json"
    
    # Assuming your dataset is in a JSON format, load it into a Hugging Face Dataset
    dataset = load_dataset("json", data_files=dataset_path, split='train')

    # Tokenize the dataset
    def tokenize_function(examples):
        return models["sentiment_tokenizer"](examples["text"], padding="max_length", truncation=True)

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Create the 'label' column from the 'sentiment' column (or 'intent_label' if needed)
    # Ensure that the sentiment is mapped to integer labels if it's not already in integer format
    tokenized_datasets = tokenized_datasets.map(lambda x: {"label": x["sentiment"]}, batched=True)

    # Ensure the dataset has input_ids and attention_mask
    tokenized_datasets.set_format(type="torch", columns=["text", "intent_label", "sentiment", "detailed_actionable_insights", "input_ids", "attention_mask", "label"])

    # Split the dataset into train and eval
    dataset = tokenized_datasets.train_test_split(test_size=0.1)

    return dataset['train'], dataset['test']

# Load and process data
train_dataset, eval_dataset = load_and_process_data()
print(f"Train dataset: {train_dataset}")
print(f"Eval dataset: {eval_dataset}")

# Fine-tune models
fine_tune_sentiment_model(train_dataset, eval_dataset)
fine_tune_intent_model(train_dataset, eval_dataset)
fine_tune_response_model(train_dataset)
fine_tune_summarization_model(train_dataset)

def analyze_sentiment(text):
    """Perform sentiment analysis using the fine-tuned model."""
    sentiment_pipeline = pipeline("sentiment-analysis", model=models["sentiment_model"], tokenizer=models["sentiment_tokenizer"])
    result = sentiment_pipeline(text)
    
    sentiment_score = result[0]["label"]
    sentiment_index = int(sentiment_score.split("_")[-1])  # Extracts number from 'LABEL_0', 'LABEL_1', etc.
    sentiment_category = SENTIMENT_LABELS.get(sentiment_index, "Unknown")

    return sentiment_category, result[0]["score"]

def recognize_intent(text):
    """Perform intent recognition using the fine-tuned model."""
    inputs = models["intent_tokenizer"](text, return_tensors="pt", truncation=True, max_length=128)
    outputs = models["intent_model"](**inputs)
    intent_scores = torch.softmax(outputs.logits, dim=1)
    predicted_intent_index = torch.argmax(intent_scores).item()
    return INTENT_LABELS.get(predicted_intent_index, "Unknown Intent")

def generate_response(text):
    """Generate a response using the fine-tuned response model."""
    inputs = models["response_tokenizer"](text, return_tensors="pt")
    outputs = models["response_model"].generate(
        inputs.input_ids,
        max_length=100,
        temperature=0.3,
        top_p=0.95,
        do_sample=True,
        pad_token_id=models["response_tokenizer"].eos_token_id,
        no_repeat_ngram_size=2
    )
    response = models["response_tokenizer"].decode(outputs[0], skip_special_tokens=True)
    return response.replace(text, "").strip()

def summarize_transcript(transcript):
    """Summarize a conversation transcript using the fine-tuned summarization model."""
    combined_text = " ".join([entry["text"] for entry in transcript])
    inputs = models["summarization_tokenizer"](combined_text, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = models["summarization_model"].generate(
        inputs.input_ids,
        max_length=150,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    summary = models["summarization_tokenizer"].decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Flask app setup
app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze text for sentiment and intent using fine-tuned models."""
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    sentiment_category, sentiment_score = analyze_sentiment(text)
    intent_label = recognize_intent(text)

    response = {
        "text": text,
        "sentiment": {
            "label": sentiment_category,
            "score": sentiment_score,
        },
        "intent": intent_label,
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=True)
