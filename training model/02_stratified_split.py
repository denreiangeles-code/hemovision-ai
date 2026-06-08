import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def execute_3way_stratified_split(input_csv="cleaned_master_index.csv"):
    print(f"Loading checkpoint: {input_csv}...")
    
    # Safety check: ensure Step 1 was actually run
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Run your extraction script first!")
        return

    # Load the cleaned data from Step 1
    final_df = pd.read_csv(input_csv)

    # Force the Hgb column to be strictly mathematical numbers. 
    # 'coerce' turns hidden text/garbage into NaN
    final_df['Hgb'] = pd.to_numeric(final_df['Hgb'], errors='coerce')
    
    # Drop any rogue rows that turned into NaN because they were actually text
    final_df = final_df.dropna(subset=['Hgb'])
    
    print("\n--- Executing 3-Way Stratified Data Split ---")
    
    # 1. Binning the continuous data (Hgb) for stratification
    # This guarantees severe anemia cases are evenly distributed across all sets
    bins = np.linspace(final_df['Hgb'].min(), final_df['Hgb'].max(), 5)
    hgb_categories = pd.qcut(final_df['Hgb'], q=4, labels=False, duplicates='drop')

    # 2. Split #1: Carve out the Test Set (The Vault - 20%)
    train_val_df, test_df, train_val_cats, _ = train_test_split(
        final_df, 
        hgb_categories, 
        test_size=0.20, 
        random_state=42, 
        stratify=hgb_categories
    )

    # 3. Split #2: Carve Validation out of the remaining 80%
    # Taking 25% of the remaining 80% gives us exactly 20% of the total dataset
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.25, 
        random_state=42, 
        stratify=train_val_cats 
    )

    print(f"Training Data (Homework): {len(train_df)} records")
    print(f"Validation Data (Practice Exam): {len(val_df)} records")
    print(f"Test Data (The Vault): {len(test_df)} records")
    
    # 4. Save the splits as physical CSV checkpoints
    print("\nSaving final pipeline checkpoints...")
    train_df.to_csv('train_set.csv', index=False)
    val_df.to_csv('val_set.csv', index=False)
    test_df.to_csv('test_set.csv', index=False)
    
    print("Success! Data is securely split and ready for OpenCV.")

if __name__ == "__main__":
    execute_3way_stratified_split()