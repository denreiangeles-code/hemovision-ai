import os
import pandas as pd

def verify_physical_files(row, base_dir="."):
    """
    Acts as the physical auditor. Checks the hard drive to ensure 
    the required image mask actually exists before approving the row.
    """
    dataset = row['Dataset']  # 'Italy' or 'India'
    number = row['Number']
    
    # Builds the expected folder path (e.g., "./Italy/1")
    folder_path = os.path.join(base_dir, dataset, str(number))
    
    # palpebral.png is the only mask guaranteed to be in every valid folder
    try:
        # Get a list of every single file inside that specific folder
        folder_contents = os.listdir(folder_path)
        
        # Check if 'palpebral' exists in any of the filenames (converted to lowercase for safety)
        has_palpebral = any('palpebral' in file_name.lower() for file_name in folder_contents)
        
        return has_palpebral
        
    except FileNotFoundError:
        # If the entire folder is missing (e.g., they skipped a number), drop the row safely
        return False

def extract_clean_and_verify(italy_excel_path, india_excel_path, base_image_dir="."):
    print("Initializing Data Extraction, Deep Cleanup, and Physical Verification...")
    
    # 1. Load the raw data
    df_italy = pd.read_excel(italy_excel_path, engine='openpyxl')
    df_india = pd.read_excel(india_excel_path, engine='openpyxl')

    # 2. Add Dataset origin
    df_italy['Dataset'] = 'Italy'
    df_india['Dataset'] = 'India'

    # 3. Merge the datasets
    df_combined = pd.concat([df_italy, df_india], ignore_index=True)
    initial_count = len(df_combined)

    # 4. Phase 1: Pandas Deep Cleanup (The Logic Filter)
    df_cleaned = df_combined.dropna(subset=['Hgb'])

    if 'Note' in df_cleaned.columns:
        df_cleaned.loc[:, 'Note'] = df_cleaned['Note'].fillna('')
        df_cleaned = df_cleaned[~df_cleaned['Note'].str.contains('not visible', case=False)]
        df_cleaned = df_cleaned.drop(columns=['Note'])

    df_cleaned.loc[:, 'Gender'] = df_cleaned['Gender'].fillna('Unknown')
    df_cleaned.loc[:, 'Age'] = df_cleaned['Age'].fillna(-1) 

    # 5. Phase 2: OS Module (The Physical Filter)
    # Apply the verify_physical_files function to every row
    # Keeps the row ONLY if the function returns True (the file exists)
    print("Running physical hard drive audit...")
    df_verified = df_cleaned[df_cleaned.apply(lambda row: verify_physical_files(row, base_image_dir), axis=1)]

    # 6. Lock in the final schema
    final_df = df_verified[['Dataset', 'Number', 'Gender', 'Age', 'Hgb']]
    
    dropped_count = initial_count - len(final_df)
    print(f"Pipeline complete. Dropped {dropped_count} invalid, incomplete, or physically missing records.")
    print(f"Total verified, training-ready patient records: {len(final_df)}")
    
    return final_df

if __name__ == "__main__":
    # File paths
    italy_path = 'Italy/Italy.xlsx'
    india_path = 'India/India.xlsx'
    
    # The directory where your 'Italy' and 'India' image folders live
    # Assuming they are in the exact same folder as this Python script
    base_directory = "." 
    
    try:
        master_data = extract_clean_and_verify(italy_path, india_path, base_directory)
        print("\n--- Verified Master Pipeline Data ---")
        print(master_data.head())
        master_data.to_csv('cleaned_master_index.csv', index=False)
    except FileNotFoundError:
        print("Error: Could not find the Excel files. Check your folder paths!")