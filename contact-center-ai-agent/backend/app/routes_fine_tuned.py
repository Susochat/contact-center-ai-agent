from flask import request, jsonify
from app import create_app

app = create_app()

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get input from the request
        user_input = request.json.get('input_text', '')

        # Validate input
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        # Load model and tokenizer from the app context
        model = app.config['MODEL']
        tokenizer = app.config['TOKENIZER']

        # Tokenize the input
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Generate the output with adjusted parameters
        outputs = model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_length=100,            # Maximum length of the generated text
            num_beams=5,               # Beam search to explore multiple options
            no_repeat_ngram_size=2,    # Prevent repetition of n-grams
            top_k=50,                  # Limit the possible next tokens to the top 50
            top_p=0.95,                # Use nucleus sampling (top-p sampling)
            temperature=0.1,           # Control randomness; lower for more deterministic output
            do_sample=True             # Enable sampling to get varied output
        )

        # Decode the generated response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({'generated_text': generated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
