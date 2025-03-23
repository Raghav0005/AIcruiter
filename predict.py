import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

def preprocess_text(text):
    """Clean and standardize text for better model understanding."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model(checkpoint_path="./interview_model/checkpoint-126"):
    """Load the trained model and tokenizer"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print(f"Model loaded successfully from {checkpoint_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_rating(text, model, tokenizer, max_length=256):
    """Predict interview rating for the given text"""
    # Preprocess the text in the same way as during training
    processed_text = preprocess_text(text)
    
    # Tokenize
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_rating = outputs.logits.item()
    
    return predicted_rating

def main():
    # Load the model
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        print("Failed to load model. Please check the path.")
        return
    
    print("\n===== AIcruiter Interview Rating System =====")
    print("Enter 'quit' to exit")
    
    while True:
        # Get user input
        text = input("\nEnter interview notes to rate (or 'quit' to exit): ")
        
        if text.lower() == 'quit':
            break
        
        if not text.strip():
            print("Please enter some text to analyze.")
            continue
        
        # Get prediction
        rating = predict_rating(text, model, tokenizer)
        
        # Display result
        print(f"\nPredicted Rating: {rating:.2f}/10")
        
        # Provide interpretation
        if rating >= 8:
            print("Interpretation: Excellent candidate, strongly consider")
        elif rating >= 6:
            print("Interpretation: Good candidate, worth considering")
        elif rating >= 4:
            print("Interpretation: Average candidate, may need additional screening")
        else:
            print("Interpretation: Below average candidate, proceed with caution")

if __name__ == "__main__":
    main()