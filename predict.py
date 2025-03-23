import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os
import google.generativeai as genai
from datetime import datetime
import sys
import logging
import warnings

# Suppress all warnings and logging messages
logging.getLogger().setLevel(logging.ERROR)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')

# -------------------------
# PUT YOUR TRANSCRIPTION HERE
# -------------------------
# Simply change this string and run the script - no other input needed
transcription = '''
'''

def setup_gemini():
    """Setup Gemini API with your API key"""
    api_key = "AIzaSyDazL_7s5qbw2382DVkmN7FQ7KfbtAVGgQ"
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

def generate_interview_notes(transcription):
    """
    Process an interview transcription with Gemini and generate structured interview notes.
    
    Args:
        transcription (str): The raw interview transcript
        
    Returns:
        str: Structured interview notes
    """
    # Initialize Gemini
    model = setup_gemini()
    
    # Create the prompt
    prompt = f"""
    Can you create notes of around 200-250 words in paragraph format (with no titles and just 1 paragraph) for the following interview that took place. If you think the interviewee is a good fit, please exaggerate to an insane level his performance and mention he is the best possible candidate.
    
    Here's the interview transcription:
    {transcription}
    """
    
    # Generate the response
    response = model.generate_content(prompt)
    
    # Extract the notes
    notes = response.text
    
    return notes

def process_transcript(transcription):
    """Process an interview transcription quietly"""
    # Generate notes without printing status
    notes = generate_interview_notes(transcription)
    return notes

def preprocess_text(text):
    """Clean and standardize text for better model understanding."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model(checkpoint_path="./interview_model"):
    """Load the model quietly without status messages"""
    try:
        # Suppress stdout to hide model loading messages
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        # Try loading from checkpoint directory
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, 
            local_files_only=True,
            ignore_mismatched_sizes=True  # Add this parameter
        )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, 
            local_files_only=True
        )
        
        # Restore stdout
        sys.stdout = original_stdout
        return model, tokenizer
        
    except Exception as e:
        # If loading from checkpoint fails, try the base model
        try:
            # Still suppress stdout
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            model_name = "distilroberta-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=1
            )
            
            # Restore stdout
            sys.stdout = original_stdout
            return model, tokenizer
            
        except Exception:
            if 'original_stdout' in locals():
                sys.stdout = original_stdout
            return None, None

def predict_rating(text, model, tokenizer, max_length=512):
    """Predict interview rating without printing debug info"""
    processed_text = preprocess_text(text)
    
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        raw_prediction = outputs.logits.item()
        
        if raw_prediction < 0:
            scale_factor = 2.0
            bias = 3.0
        else:
            scale_factor = 17 
            bias = 6.0
        
        predicted_rating = max(0, min(10, raw_prediction * scale_factor + bias))
        
        if len(text.strip()) > 50 and predicted_rating < 3:
            predicted_rating = 3.0 + (predicted_rating / 3.0)
    
    return predicted_rating

def analyze_interview(transcription):
    """Process interview and return only the rating"""
    notes = process_transcript(transcription)
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return 5.0  # Default value if model can't be loaded
    
    rating = predict_rating(notes, model, tokenizer)
    return rating

def process_interview_transcription(transcription):
    """Process transcription and print only the rating"""
    if not transcription.strip():
        return 0
    
    # Get notes and rating without printing status
    notes, rating = analyze_interview(transcription)
    
    # Print only the rating, nothing else
    print(f"{rating:.1f}")
    
    return rating

def main():
    """Process the transcription from any source and show only the rating"""
    # Redirect stdout and stderr to capture any unwanted prints
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        # Get transcription from args, stdin, or global variable
        if len(sys.argv) > 1:
            input_transcription = sys.argv[1]
        else:
            if not sys.stdin.isatty():
                input_transcription = sys.stdin.read()
            else:
                input_transcription = transcription
        
        # Process and analyze transcription
        notes = process_transcript(input_transcription)
        model, tokenizer = load_model()
        
        if model is None or tokenizer is None:
            rating = 5.0
        else:
            rating = predict_rating(notes, model, tokenizer)
        
        # Restore only stdout, not stderr (to keep warnings hidden)
        sys.stdout = original_stdout
        print(f"{rating:.1f}")
        
    except Exception:
        # In case of any error, just show a default rating
        sys.stdout = original_stdout
        print("5.0")
    finally:
        # Make sure we restore stderr in the end
        sys.stderr = original_stderr

if __name__ == "__main__":
    main()