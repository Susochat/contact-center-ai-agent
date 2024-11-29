from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the DistilGPT2 model and tokenizer for text generation
def load_distilgpt2_model():
    try:
        # Load pre-trained DistilGPT2 tokenizer and model
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Move model to CPU (since you have limited resources)
        model.to('cpu')

        return model, tokenizer
    except Exception as e:
        raise Exception(f"Error loading model or tokenizer: {str(e)}")

# Generate response using DistilGPT2
def get_distilgpt2_response(input_text, model, tokenizer):
    try:
        # Tokenize input text
        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

        # Generate output using the model
        outputs = model.generate(
            inputs,
            max_length=150,  # Adjust the max length as per your need
            num_return_sequences=1,
            temperature=0.7,  # Adjust temperature for randomness in output
            top_p=0.9,  # Use nucleus sampling for diversity
            no_repeat_ngram_size=2,  # Avoid repetition
            pad_token_id=tokenizer.eos_token_id  # Ensure padding doesn't crash
        )

        # Decode the generated text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")
