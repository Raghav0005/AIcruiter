import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# -------------------------
# 1. Enhanced Text Preprocessing
# -------------------------
def preprocess_text(text):
    """Clean and standardize text for better model understanding."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------
# 2. Improved Dataset Class
# -------------------------
class InterviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        """
        Args:
            df: A pandas DataFrame with columns "notes" and "rating".
            tokenizer: Hugging Face tokenizer.
            max_length: Maximum token length per sample.
        """
        # Preprocess texts
        self.texts = [preprocess_text(text) for text in df['notes'].tolist()]
        self.labels = df['rating'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        rating = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Remove extra batch dimension
        encoding = {key: val.squeeze() for key, val in encoding.items()}
        # Convert the rating to a float tensor
        encoding['labels'] = torch.tensor(float(rating), dtype=torch.float)
        return encoding

# -------------------------
# 3. Data Preparation
# -------------------------
df = pd.read_csv("new_augmented_data.csv")

# Check the distribution of ratings to understand the task better
print("\nRating distribution in dataset:")
print(df['rating'].describe())

# Normalize ratings if they're on different scales (optional)
# df['rating'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min()) * 10

# Balance dataset if needed (optional)
# from sklearn.utils import resample
# grouped = df.groupby('rating')
# balanced_dfs = []
# max_size = 100  # Choose appropriate size
# for _, group in grouped:
#     if len(group) > max_size:
#         balanced_dfs.append(resample(group, n_samples=max_size, random_state=42))
#     else:
#         balanced_dfs.append(group)
# df = pd.concat(balanced_dfs)

# Split with stratification if ratings are discrete
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, 
    stratify=pd.qcut(df['rating'], 5, duplicates='drop') if len(df) > 100 else None
)

# -------------------------
# 4. Advanced Model Selection
# -------------------------
# Choose a more powerful model (uncomment one)
# model_name = "microsoft/deberta-v3-base"  # Superior performance
# model_name = "facebook/bart-large"      # Good for text understanding
# model_name = "roberta-large"            # Larger version of RoBERTa
model_name = "distilroberta-base"       # Faster, slightly less accurate

print(f"\nUsing model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# -------------------------
# 5. Create Enhanced Datasets
# -------------------------
# Increase max_length to capture more context
train_dataset = InterviewDataset(train_df, tokenizer, max_length=512)
val_dataset = InterviewDataset(val_df, tokenizer, max_length=512)

# ------------------------------------------------
# 5. Define Metrics for Trainer's "compute_metrics"
# ------------------------------------------------
def compute_metrics(eval_pred):
    """
    Hugging Face Trainer will pass (predictions, label_ids) to this function at eval time.
    We'll compute standard regression metrics: MSE, MAE, and R².
    """
    predictions, labels = eval_pred
    # predictions is shape (batch_size, 1), so flatten to compare with labels
    predictions = predictions.flatten()
    labels = labels.flatten()

    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# -------------------------
# 6. Optimized Training Arguments
# -------------------------
training_args = TrainingArguments(
    output_dir="./interview_model",
    # Longer training for better performance
    num_train_epochs=8,
    # Use larger batch sizes if your GPU allows
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # Gradient accumulation simulates larger batch sizes
    gradient_accumulation_steps=4,
    # Evaluation and checkpointing strategy
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    # Optimized learning parameters
    learning_rate=1e-5,  # Lower learning rate for more stable training
    warmup_ratio=0.1,
    weight_decay=0.01,
    # Prevent overfitting
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # For reproducibility
    seed=42,
    # Efficiency improvements if you have compatible hardware
    fp16=True if torch.cuda.is_available() else False,
)

# -------------------------
# 7. Enhanced Training Setup
# -------------------------
# Use a data collator for dynamic padding (more efficient)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Import the callback
from transformers import EarlyStoppingCallback

# Add early stopping callback when creating the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add this line
)

# -------------------------
# 8. Model Training
# -------------------------
print("\nTraining model with optimized parameters...")
trainer.train()

# -------------------------
# 9. Comprehensive Evaluation
# -------------------------
print("\nEvaluating model performance...")

# Get predictions
model.eval()
val_predictions = []
val_true_labels = []

for i in range(len(val_dataset)):
    item = val_dataset[i]
    inputs = {
        'input_ids': item['input_ids'].unsqueeze(0),
        'attention_mask': item['attention_mask'].unsqueeze(0)
    }
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_rating = outputs.logits.item()
    true_rating = item['labels'].item()

    val_predictions.append(predicted_rating)
    val_true_labels.append(true_rating)

# Calculate metrics
mse = mean_squared_error(val_true_labels, val_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(val_true_labels, val_predictions)
r2 = r2_score(val_true_labels, val_predictions)

print("\nEvaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Multiple accuracy thresholds for a better understanding
thresholds = [0.5, 0.75, 1.0, 1.5, 2.0]
for threshold in thresholds:
    within_threshold = sum(abs(np.array(val_predictions) - np.array(val_true_labels)) <= threshold)
    accuracy = within_threshold / len(val_true_labels) * 100
    print(f"Accuracy (predictions within {threshold} points): {accuracy:.2f}%")

# -------------------------
# 10. Visualize Results
# -------------------------
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.scatter(val_true_labels, val_predictions, alpha=0.5)
    # Add diagonal line representing perfect predictions
    min_val = min(min(val_true_labels), min(val_predictions))
    max_val = max(max(val_true_labels), max(val_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add threshold bounds
    threshold = 1.0  # 1-point threshold
    plt.fill_between(
        [min_val, max_val], 
        [min_val - threshold, max_val - threshold], 
        [min_val + threshold, max_val + threshold], 
        alpha=0.2, color='green'
    )
    
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Prediction Performance')
    plt.savefig('prediction_performance.png')
    print("\nVisualization saved as 'prediction_performance.png'")
except ImportError:
    print("\nMatplotlib not available for visualization")

# -------------------------
# 11. Sample Inference
# -------------------------
sample_note = "The candidate demonstrated excellent problem-solving skills and a positive attitude."
inputs = tokenizer(sample_note, return_tensors="pt", truncation=True, 
                   padding="max_length", max_length=256)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predicted_rating = outputs.logits.item()
    print(f"\nSample Prediction:")
    print(f"Predicted interview rating: {predicted_rating:.2f} out of 10")