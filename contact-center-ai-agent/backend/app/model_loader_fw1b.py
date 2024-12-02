from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_falcon_rw_1b_model():
    try:
        model_name = "tiiuae/falcon-rw-1b"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model without quantization (on CPU)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,  # Required for Falcon models
            device_map="cpu",        # Ensure CPU usage
            # low_cpu_mem_usage=True # Optimize for CPU usage
        )

        # Set model to use CPU explicitly
        model = model.to("cpu")

        return model, tokenizer
    except Exception as e:
        raise Exception(f"Error loading Falcon-RW-1B model: {str(e)}")


def get_falcon_rw_1b_response(input_text, model, tokenizer):
    """
    Generate a focused response using the Falcon-RW-1B model.
    """
    try:
        # Tokenize the input text
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Generate a response using the model
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,             # Set max length for a concise response
            temperature=0.2,            # Lower randomness for more focused responses
            top_p=0.9,                  # Nucleus sampling for diversity
            top_k=30,                   # Cap the number of highest probability candidates
            no_repeat_ngram_size=2,     # Avoid repeated phrases
            pad_token_id=tokenizer.eos_token_id,  # Ensure proper padding
            eos_token_id=tokenizer.eos_token_id,  # Use EOS token for stopping generation
            early_stopping=True         # Stop when model is done generating
        )

        # Decode the output and return a cleaner response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure a more natural response (removing excessive repetitions, etc.)
        response = response.strip()

        return response

    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")
