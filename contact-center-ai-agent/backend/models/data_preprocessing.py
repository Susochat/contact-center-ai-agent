from transformers import DistilGPT2Tokenizer
import pandas as pd
from torch.utils.data import Dataset

# Load the DistilGPT2 tokenizer
tokenizer = DistilGPT2Tokenizer.from_pretrained('distilgpt2')

# Ensure padding token is defined since GPT models typically don't have one by default
tokenizer.pad_token = tokenizer.eos_token

# Load your custom dataset (replace with the actual path)
data_path = "C:\\Users\\susab\\OneDrive\\Desktop\\CC-AI-Agent\\contact-center-ai-agent\\backend\\data\\contact_center_synthetic_dataset.json"
df = pd.read_json(data_path)

# Create a Dataset class to tokenize inputs and outputs
class ContactCenterDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Input and output pair
        input_text = self.data.iloc[idx]["input"]
        output_text = self.data.iloc[idx]["output"]

        # Tokenize input and output
        input_encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        output_encoding = self.tokenizer.encode_plus(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Flatten the tensors to avoid extra dimension
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': output_encoding['input_ids'].squeeze(0)
        }

# Create the dataset and dataloader
dataset = ContactCenterDataset(df, tokenizer)

# Check if everything works
print(dataset[0])  # Print the first example to verify
