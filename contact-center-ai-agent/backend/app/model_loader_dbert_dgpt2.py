from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BartForConditionalGeneration, BartTokenizer, pipeline
import torch

# Define human-readable broader intent labels
INTENT_LABELS = {
    0: "Order Management",        # Includes: Order Status, Request for Replacement, Return Process, Order Modification
    1: "Technical Support",       # Includes: Technical Support, Technical Assistance, Service Cancellation
    2: "Account & Billing",       # Includes: Account Access, Billing Inquiry, Payment Issues, Account Update
    3: "Product Inquiry",         # Includes: Product Inquiry, Product Availability, Promotions, Warranty Information
    4: "General Inquiry & Feedback"  # Includes: Shipping Inquiry, Feedback, Complaint, Subscription Issue, General Inquiry
}

# Load models and tokenizers
def load_models():
    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

    intent_model_name = "distilbert-base-uncased"  # Placeholder for intent recognition fine-tuned model
    intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name)
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

def analyze_sentiment(text):
    """Perform sentiment analysis."""
    sentiment_pipeline = pipeline("sentiment-analysis", model=models["sentiment_model"], tokenizer=models["sentiment_tokenizer"])
    result = sentiment_pipeline(text)
    return result[0]["label"], result[0]["score"]

def recognize_intent(text):
    """Perform intent recognition and map to human-readable broader intent labels."""
    inputs = models["intent_tokenizer"](text, return_tensors="pt", truncation=True, max_length=128)
    outputs = models["intent_model"](**inputs)
    intent_scores = torch.softmax(outputs.logits, dim=1)
    predicted_intent_index = torch.argmax(intent_scores).item()
    return INTENT_LABELS.get(predicted_intent_index, "Unknown Intent")

def generate_response(text):
    """Generate a response using DistilGPT2 without redundancy."""
    inputs = models["response_tokenizer"](text, return_tensors="pt")
    outputs = models["response_model"].generate(
        inputs.input_ids,
        max_length=100,
        temperature=0.3,  # Increased temperature for more diverse responses
        top_p=0.95,  # Ensuring more randomness in the response generation
        do_sample=True,
        pad_token_id=models["response_tokenizer"].eos_token_id,
        no_repeat_ngram_size=2  # Prevents repeating the same n-grams (e.g., sequences of 2 words)
    )
    response = models["response_tokenizer"].decode(outputs[0], skip_special_tokens=True)
    return response.replace(text, "").strip()

def summarize_transcript(transcript):
    """Summarize a conversation transcript."""
    combined_text = " ".join([entry["text"] for entry in transcript])
    inputs = models["summarization_tokenizer"](combined_text, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = models["summarization_model"].generate(
        inputs.input_ids,
        max_length=150,  # Set the maximum summary length
        num_beams=4,  # Increase the number of beams for better quality
        early_stopping=True,
        no_repeat_ngram_size=2  # Prevents repeating the same n-grams (e.g., sequences of 2 words)
    )
    summary = models["summarization_tokenizer"].decode(summary_ids[0], skip_special_tokens=True)
    return summary
