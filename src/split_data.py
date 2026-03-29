import splitfolders
import os

# Path to your raw Kaggle images
input_folder = 'data/raw/Training' 

if os.path.exists(input_folder):
    # Splits into 70% Train, 20% Val, 10% Test
    splitfolders.ratio(input_folder, output="data/processed", 
                       seed=1337, ratio=(.7, .2, .1), group_prefix=None)
    print("✅ Data split complete! Check the 'data/processed' folder.")
else:
    print(f"❌ Error: Could not find {input_folder}. Check your folder names!")