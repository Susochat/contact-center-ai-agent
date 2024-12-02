from flask import Flask, request, jsonify
from app.model_loader_dbert_dgpt2 import load_models, analyze_sentiment, recognize_intent, generate_response, summarize_transcript

# Initialize Flask app
app = Flask(__name__)


def init_routes(app):
    """Initialize all routes for the Flask app."""

    @app.route("/analyze", methods=["POST"])
    def analyze():
        """Analyze the customer input."""
        try:
            input_data = request.json
            if not input_data:
                return jsonify({"error": "No input data provided."}), 400

            # If input is a transcript
            if "conversation_id" in input_data and "transcript" in input_data:
                transcript = input_data["transcript"]
                summary = summarize_transcript(transcript)
                sentiment_results = [analyze_sentiment(entry["text"]) for entry in transcript]
                intents = [recognize_intent(entry["text"]) for entry in transcript if entry["speaker"] == "Customer"]

                return jsonify({
                    "conversation_id": input_data["conversation_id"],
                    "summary": summary,
                    "sentiments": [{"text": entry["text"], "sentiment": sentiment, "score": score} for entry, (sentiment, score) in zip(transcript, sentiment_results)],
                    "intents": [recognize_intent(intent) for intent in intents]
                }), 200

            # If input is standalone text
            elif "input_text" in input_data:
                user_input = input_data["input_text"]
                sentiment, sentiment_score = analyze_sentiment(user_input)
                intent = recognize_intent(user_input)
                response = generate_response(user_input)

                return jsonify({
                    "sentiment": {"label": sentiment, "score": sentiment_score},
                    "intent": intent,
                    "response": response,
                }), 200

            else:
                return jsonify({"error": "Invalid input format."}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/", methods=["GET"])
    def home():
        """Home route to check server status."""
        return jsonify({"message": "Contact Center AI Agent is running."}), 200