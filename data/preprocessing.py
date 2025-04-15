import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
if not os.path.exists("datasets/raw/combined_games.csv"):
    utils.combine_datasets()
df = pd.read_csv("datasets/raw/combined_games.csv")

# Move reviewScore to the last column if it exists
if 'reviewScore' in df.columns:
    # Get all columns except reviewScore
    cols = [col for col in df.columns if col != 'reviewScore']
    # Add reviewScore at the end
    cols.append('reviewScore')
    # Reorder columns
    df = df[cols]
    # Save the reordered DataFrame back to CSV
    df.to_csv("datasets/raw/combined_games.csv", index=False)

print(df.columns)
print(df.isna().sum())

