from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the DistilGPT2 model and tokenizer for text generation
def load_distilgpt2_model():
    """
    Loads the DistilGPT2 model and tokenizer.

    Returns:
        model (AutoModelForCausalLM): Pre-trained DistilGPT2 model.
        tokenizer (AutoTokenizer): Pre-trained tokenizer with a set padding token.
    Raises:
        Exception: If there is an error loading the model or tokenizer.
    """
    model_name = "distilgpt2"
    try:
        # Load pre-trained DistilGPT2 tokenizer and model
        print(f"Loading model and tokenizer for '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Set the padding token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Ensure model is loaded on the appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model loaded successfully on device: {device}")

        return model, tokenizer
    except Exception as e:
        raise Exception(f"Error loading model or tokenizer: {str(e)}")


# Generate response using DistilGPT2
def get_distilgpt2_response(input_text, model, tokenizer):
    """
    Generates a response using the DistilGPT2 model.

    Args:
        input_text (str): The user input text.
        model (AutoModelForCausalLM): Pre-trained DistilGPT2 model.
        tokenizer (AutoTokenizer): Tokenizer for DistilGPT2.

    Returns:
        str: Generated response.
    Raises:
        Exception: If there is an error during response generation.
    """
    try:
        if not input_text.strip():
            raise ValueError("Input text cannot be empty.")

        # Determine the device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Tokenize input text
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        # Generate output using the model
        outputs = model.generate(
            inputs["input_ids"],
            max_length=150,  # Adjust as needed
            num_return_sequences=1,
            temperature=0.1,  # Controls randomness
            top_p=0.8,  # Nucleus sampling for diversity
            do_sample=True,
            no_repeat_ngram_size=2,  # Avoid repetition
            pad_token_id=tokenizer.pad_token_id,  # Use the explicitly set padding token
            attention_mask=inputs["attention_mask"]  # Support for attention masks
        )

        # Decode the generated text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove input query from the response
        if response.lower().startswith(input_text.lower()):
            response = response[len(input_text):].strip()

        return response.strip()

    except ValueError as ve:
        raise ValueError(f"Validation Error: {str(ve)}")
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")
