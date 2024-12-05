from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
    Trainer,
    TrainingArguments
)
import torch
from datasets import load_dataset
from flask_cors import CORS  # Import CORS


# Define human-readable broader intent labels (can be used for reference or mapping)
INTENT_LABELS = {
    0: "Order Management",        # Includes: Order Status, Request for Replacement, Return Process, Order Modification
    1: "Technical Support",       # Includes: Technical Support, Technical Assistance, Service Cancellation
    2: "Account & Billing",       # Includes: Account Access, Billing Inquiry, Payment Issues, Account Update
    3: "Product Inquiry",         # Includes: Product Inquiry, Product Availability, Promotions, Warranty Information
    4: "General Inquiry & Feedback"  # Includes: Shipping Inquiry, Feedback, Complaint, Subscription Issue, General Inquiry
}


# Load models and tokenizers
def load_models():
    # Load Sentiment Analysis Model and Tokenizer
    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

    # Load Intent Recognition Model and Tokenizer
    intent_model_name = "distilbert-base-uncased"  # Placeholder for intent recognition fine-tuned model
    intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name)
    intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)

    # Load GPT-2 model for response generation (actionable steps generation)
    response_model_name = "distilgpt2"
    response_model = AutoModelForCausalLM.from_pretrained(response_model_name)
    
    # Ensure that add_prefix_space is set to True for GPT-2 tokenizer
    response_tokenizer = AutoTokenizer.from_pretrained(response_model_name, use_fast=True, add_prefix_space=True)
    response_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    response_tokenizer.pad_token = response_tokenizer.eos_token  # Set padding token to EOS token

    # Load Summarization Model and Tokenizer (if needed for any future tasks)
    summarization_model_name = "facebook/bart-large-cnn"
    summarization_model = AutoModelForCausalLM.from_pretrained(summarization_model_name)
    summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)

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


def analyze_sentiment(text):
    """Perform sentiment analysis."""
    sentiment_pipeline = pipeline("sentiment-analysis", model=models["sentiment_model"], tokenizer=models["sentiment_tokenizer"])
    result = sentiment_pipeline(text)
    return result[0]["label"], result[0]["score"]


def recognize_intent(text):
    """Recognize intent based on the input text."""
    inputs = models["intent_tokenizer"](text, return_tensors="pt", truncation=True, max_length=128)
    outputs = models["intent_model"](**inputs)
    intent_scores = torch.softmax(outputs.logits, dim=1)
    predicted_intent_index = torch.argmax(intent_scores).item()
    return INTENT_LABELS.get(predicted_intent_index, "Unknown Intent")


def generate_response(text):
    """Generate actionable steps (response) based on the input text."""
    inputs = models["response_tokenizer"](text, return_tensors="pt", padding=True, truncation=True)
    outputs = models["response_model"].generate(
        inputs.input_ids,
        max_length=150,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        pad_token_id=models["response_tokenizer"].pad_token_id,
        no_repeat_ngram_size=2
    )
    response = models["response_tokenizer"].decode(outputs[0], skip_special_tokens=True)
    return response.replace(text, "").strip()


# Load and process dataset
def load_and_process_data():
    # Load dataset from JSON file
    dataset_path = "../data/contact_center_processed_dataset.json"  # Replace with your actual file path
    dataset = load_dataset("json", data_files=dataset_path, split='train')

    # Tokenize the dataset for fine-tuning
    def tokenize_function(examples):
        return models["response_tokenizer"](examples["text"], padding="max_length", truncation=True)

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Set the dataset format for PyTorch
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "text", "detailed_actionable_insights"])

    # Split the dataset into train and evaluation
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)

    return tokenized_datasets['train'], tokenized_datasets['test']


train_dataset, eval_dataset = load_and_process_data()
print(f"Train dataset: {train_dataset}")
print(f"Eval dataset: {eval_dataset}")


# Fine-tune the model with the dataset
def fine_tune_model():
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
        eval_dataset=eval_dataset,
        tokenizer=models["response_tokenizer"]
    )

    trainer.train()



# Flask app setup
app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze input text to generate actionable steps."""
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    sentiment_label, sentiment_score = analyze_sentiment(text)
    intent_label = recognize_intent(text)
    response = generate_response(text)

    response_data = {
        "text": text,
        "sentiment": {
            "label": sentiment_label,
            "score": sentiment_score,
        },
        "intent": intent_label,
        "actions": response
    }

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=True)
