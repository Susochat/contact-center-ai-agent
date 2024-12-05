import json
import pandas as pd
import matplotlib.pyplot as plt

# File paths
INPUT_PATH = "../data/contact_center_ai_agent_dataset.json"  # Replace with your dataset file path
OUTPUT_PATH_CATEGORY_DIST = "../data/category_distribution.png"
OUTPUT_PATH_SENTIMENT_DIST = "../data/sentiment_distribution.png"
OUTPUT_PATH_CONV_LEN = "../data/conversation_length.png"

# Load the dataset
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert dataset to a pandas DataFrame
rows = []
for entry in data:
    intent_label = entry["intent_label"]
    sentiment = entry["sentiment"]
    insights_length = len(entry["detailed_actionable_insights"])  # Use actionable insights as a proxy for detail
    rows.append({
        "Intent_Label": intent_label,
        "Sentiment": sentiment,
        "Text": entry["text"],
        "Insights_Length": insights_length
    })

df = pd.DataFrame(rows)

# Analyze and visualize the data
# 1. Distribution of intent labels
intent_counts = df["Intent_Label"].value_counts()
intent_counts.plot(kind="bar", title="Intent Label Distribution", color="skyblue")
plt.xlabel("Intent Label")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_PATH_CATEGORY_DIST)
plt.show()

# 2. Distribution of sentiments
sentiment_counts = df["Sentiment"].value_counts()
sentiment_counts.plot(kind="pie", title="Sentiment Distribution", autopct="%1.1f%%", colors=["lightgreen", "lightcoral", "gold"])
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_PATH_SENTIMENT_DIST)
plt.show()

# 3. Insights length analysis
insights_lengths = df.groupby("Intent_Label")["Insights_Length"].mean()
insights_lengths.plot(kind="bar", title="Average Number of Insights by Intent Label", color="orange")
plt.xlabel("Intent Label")
plt.ylabel("Average Number of Insights")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_PATH_CONV_LEN)
plt.show()

print("Dataset analysis completed. Visualizations saved.")
