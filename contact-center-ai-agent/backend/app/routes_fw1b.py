from flask import Flask, request, jsonify
from app.model_loader_fw1b import load_falcon_rw_1b_model

# Initialize Flask app
app = Flask(__name__)

# Load the Falcon-RW-1B model and tokenizer during app initialization
print("Loading Falcon-RW-1B model and tokenizer...")
model, tokenizer = load_falcon_rw_1b_model()
print("Model and tokenizer loaded successfully!")

# Solution to set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token if not defined
    print("Padding token has been set to eos_token.")

def init_routes(app):
    """Initialize all routes for the Flask app."""

    @app.route("/", methods=["GET"])
    def home():
        """Home route to check server status."""
        return jsonify({"message": "Contact Center AI Agent is running."}), 200

    @app.route("/generate", methods=["POST"])
    def generate_response():
        """Route to handle text generation requests."""
        try:
            # Parse the input JSON payload
            input_data = request.json
            if not input_data or "input_text" not in input_data:
                return jsonify({"error": "No input_text provided."}), 400

            user_input = input_data["input_text"]
            print(f"Received input: {user_input}")

            # Tokenize the input text
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                max_length=256,  # Limit input length for performance
                padding=True,
                return_attention_mask=True
            )

            # Generate response using the Falcon-RW-1B model
            response = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,  # Enable sampling
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id
            )

            # Decode the generated response
            generated_response = tokenizer.decode(response[0], skip_special_tokens=True)

            # Remove the user input from the generated response if it repeats
            if generated_response.lower().startswith(user_input.lower()):
                generated_response = generated_response[len(user_input):].strip()

            return jsonify({"response": generated_response.strip()}), 200

        except Exception as e:
            # Handle errors gracefully
            return jsonify({"error": str(e)}), 500


