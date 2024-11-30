import json

# Load the dataset
file_path = "../data/contact_center_actionable_dataset.json"  # Replace with your dataset file path
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 1: Remove duplicate rows
unique_data = []
seen_conversations = set()

for entry in data:
    conversation_tuple = tuple(
        (turn["role"], turn["text"].strip().lower()) for turn in entry["conversation"]
    )
    if conversation_tuple not in seen_conversations:
        seen_conversations.add(conversation_tuple)
        unique_data.append(entry)

# Step 2: Check for missing values and filter incomplete entries
cleaned_data = []
for entry in unique_data:
    if (
        "category" in entry
        and "sentiment" in entry
        and "conversation" in entry
        and all(
            "role" in turn and "text" in turn and turn["text"].strip()
            for turn in entry["conversation"]
        )
    ):
        entry["category"] = entry["category"].strip()
        entry["sentiment"] = entry["sentiment"].strip()
        # Normalize text
        for turn in entry["conversation"]:
            turn["text"] = turn["text"].strip().lower()
        cleaned_data.append(entry)

# Save the cleaned dataset
output_path = "../data/contact_center_processed_dataset.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

print(f"Cleaned dataset saved to {output_path}")
