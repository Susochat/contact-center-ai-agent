from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Flask app
app = Flask(__name__)

# Function to load the SalesGPT model and tokenizer
def load_salesgpt_model():
    """
    Load the SalesGPT model and tokenizer with the specified configurations.
    """
    model_name = "mariordoniez/sales_updated"
    tokenizer_name = "microsoft/phi-1_5"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"  # Use CPU for loading and inference
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True
    )
    return model, tokenizer

# Load the model and tokenizer during app startup
model, tokenizer = load_salesgpt_model()
print("SalesGPT_v2 model and tokenizer loaded successfully!")

# Function to generate a response
def generate_response(input_text, model, tokenizer):
    """
    Generate a response using the SalesGPT model.
    """
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        inputs.input_ids,
        max_length=200,  # Maximum response length
        temperature=0.7,  # Adjust randomness
        top_p=0.9,  # Nucleus sampling
        do_sample=True,  # Enable sampling for diversity
        pad_token_id=tokenizer.eos_token_id  # Set padding token
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Define Flask routes
@app.route("/", methods=["GET"])
def home():
    """Check server status."""
    return jsonify({"message": "SalesGPT Contact Center AI is running."}), 200

@app.route("/generate", methods=["POST"])
def generate():
    """Handle POST request for generating a response."""
    try:
        input_data = request.json
        if not input_data or "input_text" not in input_data:
            return jsonify({"error": "No input_text provided."}), 400
        
        user_input = input_data["input_text"]
        print(f"Received input: {user_input}")

        # Generate response
        response = generate_response(user_input, model, tokenizer)
        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
