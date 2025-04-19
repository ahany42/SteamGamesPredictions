import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
if not os.path.exists("datasets/raw/combined_games.csv"):
    utils.combine_datasets()
df = pd.read_csv("datasets/raw/combined_games.csv")


def EDA(df):
    pass
def preprocess(df):
    pass

print(df.columns)
print(df.isna().sum())

