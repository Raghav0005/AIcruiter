import pandas as pd
import numpy as np
import nltk
import random
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split

nltk.download('wordnet')

def get_synonyms(word):
    """Get synonyms for a word using WordNet"""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word and "_" not in lemma.name():
                synonyms.append(lemma.name())
    return list(set(synonyms))

def synonym_replacement(text, n=2):
    """Replace n words in the text with their synonyms"""
    words = text.split()
    # Filter out words with no synonyms
    replaceable_words = [word for word in words if len(get_synonyms(word)) > 0]
    
    # If we have fewer replaceable words than n, adjust n
    n = min(n, len(replaceable_words))
    
    if n == 0:
        return text
        
    # Randomly select n words to replace
    to_replace = random.sample(replaceable_words, n)
    
    # Replace each selected word with a random synonym
    for word in to_replace:
        synonyms = get_synonyms(word)
        if synonyms:
            replacement = random.choice(synonyms)
            text = text.replace(" " + word + " ", " " + replacement + " ")
    
    return text

def augment_data(df, augment_factor=2):
    """Augment the dataset by creating variations of existing texts"""
    original_len = len(df)
    new_rows = []
    
    # Check if 'id' column exists, if not create one
    if 'id' not in df.columns:
        df['id'] = [f"orig_{i}" for i in range(len(df))]
        print("Added ID column to original data")
    
    # Get the highest numeric part of IDs to continue numbering
    last_id = 0
    for id_val in df['id']:
        if isinstance(id_val, str) and "_" in id_val:
            try:
                num_part = int(id_val.split("_")[1])
                last_id = max(last_id, num_part)
            except (ValueError, IndexError):
                pass
        elif isinstance(id_val, (int, float)):
            last_id = max(last_id, int(id_val))
    
    next_id = last_id + 1
    
    for _ in range(augment_factor - 1):
        for i, row in df.iterrows():
            text = row['notes']
            rating = row['rating']
            orig_id = row['id']
            
            # Create an augmented version
            augmented_text = synonym_replacement(text, n=int(len(text.split()) * 0.1))
            
            # Add slight noise to the rating (optional)
            # augmented_rating = max(0, min(10, rating + random.uniform(-0.3, 0.3)))
            augmented_rating = rating  # Or keep the same rating
            
            # Create a new ID that references the original
            augmented_id = f"{next_id}"
            next_id += 1
            
            new_rows.append({
                'id': augmented_id,
                'notes': augmented_text, 
                'rating': augmented_rating,
                'derived_from': orig_id  # Optional: track source ID
            })
    
    # Create a dataframe from the new rows and concatenate with original
    augmented_df = pd.DataFrame(new_rows)
    
    # If we added a 'derived_from' column to augmented data but it doesn't exist in original
    if 'derived_from' in augmented_df.columns and 'derived_from' not in df.columns:
        df['derived_from'] = df['id']  # Original data derives from itself
    
    final_df = pd.concat([df, augmented_df], ignore_index=True)
    
    print(f"Original dataset size: {original_len}")
    print(f"Augmented dataset size: {len(final_df)}")
    
    return final_df

if __name__ == "__main__":
    # Load the original data
    df = pd.read_csv("data.csv")
    
    # Add ID column if it doesn't exist
    if 'id' not in df.columns:
        df['id'] = [f"orig_{i}" for i in range(len(df))]
        print("Added ID column to original data")
    
    # Perform augmentation
    augmented_df = augment_data(df, augment_factor=3)
    
    # Ensure ID column is first in the CSV
    column_order = ['id'] + [col for col in augmented_df.columns if col != 'id']
    augmented_df = augmented_df[column_order]
    
    # Save the augmented dataset
    augmented_df.to_csv("new_augmented_data.csv", index=False)
    print("Augmented data saved to 'augmented_data.csv'")
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(augmented_df, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Save train and val sets with IDs preserved
    train_df.to_csv("train_data.csv", index=False)
    val_df.to_csv("val_data.csv", index=False)
    print("Split data saved to 'train_data.csv' and 'val_data.csv'")