from flask import Flask, request, jsonify
from app.model_loader import load_distilgpt2_model, get_distilgpt2_response

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer during app initialization
print("Loading DistilGPT2 model and tokenizer...")
model, tokenizer = load_distilgpt2_model()
print("Model and tokenizer loaded successfully!")

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

            # Generate response using the model
            response = get_distilgpt2_response(user_input, model, tokenizer)

            # Return the generated response
            return jsonify({"response": response}), 200

        except Exception as e:
            # Handle errors gracefully
            return jsonify({"error": str(e)}), 500

   