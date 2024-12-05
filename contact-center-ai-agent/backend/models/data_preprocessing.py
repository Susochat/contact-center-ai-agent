import json

# File paths
INPUT_PATH = "../data/contact_center_ai_agent_dataset.json"  # Replace with your dataset file path
OUTPUT_PATH = "../data/contact_center_processed_dataset.json"

def load_dataset(file_path):
    """Load the dataset from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def remove_duplicates(data):
    """Remove duplicate rows from the dataset."""
    unique_data = []
    seen_texts = set()

    for entry in data:
        text_lower = entry["text"].strip().lower()
        if text_lower not in seen_texts:
            seen_texts.add(text_lower)
            unique_data.append(entry)
    return unique_data

def clean_and_validate_data(data):
    """Check for missing values and filter incomplete entries."""
    cleaned_data = []
    for entry in data:
        if all([
            "text" in entry and entry["text"].strip(),
            "intent_label" in entry and entry["intent_label"].strip(),
            "sentiment" in entry and entry["sentiment"].strip(),
            "detailed_actionable_insights" in entry and isinstance(entry["detailed_actionable_insights"], list)
        ]):
            entry["text"] = entry["text"].strip()
            entry["intent_label"] = entry["intent_label"].strip()
            entry["sentiment"] = entry["sentiment"].strip()
            entry["detailed_actionable_insights"] = [
                insight.strip() for insight in entry["detailed_actionable_insights"] if insight.strip()
            ]
            if entry["detailed_actionable_insights"]:  # Ensure insights are not empty
                cleaned_data.append(entry)
    return cleaned_data

def save_dataset(data, file_path):
    """Save the cleaned dataset to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Cleaned dataset saved to {file_path}")

# Processing steps
if __name__ == "__main__":
    dataset = load_dataset(INPUT_PATH)
    unique_data = remove_duplicates(dataset)
    cleaned_data = clean_and_validate_data(unique_data)
    save_dataset(cleaned_data, OUTPUT_PATH)
