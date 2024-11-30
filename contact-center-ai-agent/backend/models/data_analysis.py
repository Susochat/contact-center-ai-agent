import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the processed dataset
file_path = "../data/contact_center_processed_dataset.json"  # Ensure this matches your saved dataset
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert dataset to a pandas DataFrame
rows = []
for entry in data:
    category = entry["category"]
    sentiment = entry["sentiment"]
    conversation_length = len(entry["conversation"])
    for turn in entry["conversation"]:
        rows.append({
            "Category": category,
            "Sentiment": sentiment,
            "Role": turn["role"],
            "Text": turn["text"],
            "Conversation_Length": conversation_length
        })

df = pd.DataFrame(rows)

# Analyze and visualize the data
# 1. Distribution of categories
category_counts = df["Category"].value_counts()
category_counts.plot(kind="bar", title="Category Distribution", color="skyblue")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../data/category_distribution.png")
plt.show()

# 2. Distribution of sentiments
sentiment_counts = df["Sentiment"].value_counts()
sentiment_counts.plot(kind="pie", title="Sentiment Distribution", autopct="%1.1f%%", colors=["lightgreen", "lightcoral", "gold"])
plt.ylabel("")
plt.tight_layout()
plt.savefig("../data/sentiment_distribution.png")
plt.show()

# 3. Conversation length analysis
conversation_lengths = df.groupby("Category")["Conversation_Length"].mean()
conversation_lengths.plot(kind="bar", title="Average Conversation Length by Category", color="orange")
plt.xlabel("Category")
plt.ylabel("Average Conversation Length")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../data/conversation_length.png")
plt.show()

print("Dataset analysis completed. Visualizations saved.")
