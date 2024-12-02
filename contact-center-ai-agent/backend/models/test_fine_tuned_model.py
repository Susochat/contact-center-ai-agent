from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

# Ensure the model is in evaluation mode
model.eval()

# Test input (you can change this to a sample conversation from your dataset)
test_input = "i need assistance with billing issue?"

 # Ensure padding token is set
tokenizer.pad_token = tokenizer.eos_token 

# Tokenize the input
inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Generate the output
with torch.no_grad():
    outputs = model.generate(input_ids=inputs['input_ids'], 
                             attention_mask=inputs['attention_mask'], 
                             max_length=150, 
                             num_return_sequences=1, 
                             no_repeat_ngram_size=2, 
                             do_sample=True,
                             temperature=0.3,
                             top_p=0.9,
                             top_k=50)

# Decode the generated response
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Remove input query from the response
if generated_text.lower().startswith(test_input.lower()):
    generated_text = generated_text[len(test_input):].strip()

print("Input: ", test_input)
print("Generated Response: ", generated_text)
