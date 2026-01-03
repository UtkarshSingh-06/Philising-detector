import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

# Load dataset
df = pd.read_csv("data/phishing.csv")

# Make sure 'url' and 'Result' columns are present
if not {'url', 'Result'}.issubset(df.columns):
    raise ValueError("❌ 'url' and 'Result' columns are required!")

# Prepare labels: convert -1 to 0 for binary classification
df['label'] = df['Result'].replace(-1, 0)
df = df[['url', 'label']]  # Only keep necessary columns

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load URL-BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("url-binet/url-bert-base")

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["url"], truncation=True, padding="max_length", max_length=128)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unused columns
train_dataset = train_dataset.remove_columns(['url', 'Result']) if 'Result' in train_dataset.column_names else train_dataset
test_dataset = test_dataset.remove_columns(['url', 'Result']) if 'Result' in test_dataset.column_names else test_dataset

# Set format for PyTorch
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Load pretrained URL-BERT model
model = AutoModelForSequenceClassification.from_pretrained("url-binet/url-bert-base", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/urlbert_phishing",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Accuracy function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"\n✅ Final Test Accuracy: {eval_results['eval_accuracy']:.4f}")

# Save model and tokenizer
model.save_pretrained("./models/urlbert_phishing")
tokenizer.save_pretrained("./models/urlbert_phishing")
print("✅ Transformer model and tokenizer saved.")