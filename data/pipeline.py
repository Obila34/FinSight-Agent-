import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
   logger.info(f"Loading data from {file_path}")
   df  = pd.read_csv(file_path)
   logger.info(f"Loaded {len(df)} records from {file_path}")
   return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")

    # Fix dates
    df['date'] = pd.to_datetime(df['date'])

    # Standardise merchant names
    df['merchant'] = df['merchant'].str.strip().str.title()

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicates")

    # Remove null/zero amounts
    df = df[df['amount'].notna()]
    df = df[df['amount'] > 0]

    # Add time features
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')
    df['day_of_week'] = df['date'].dt.day_name()
    df['week'] = df['date'].dt.isocalendar().week.astype(int)

    logger.info(f"Clean dataset: {len(df)} rows")
    return df

def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)
    logger.info(f"Saved cleaned data to {filepath}")

def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_data(input_path)
    df = clean_data(df)
    save_data(df, output_path)
    return df

if __name__ == "__main__":
    df = run_pipeline(
        input_path="data/transactions.csv",
        output_path="data/cleaned_transactions.csv"
    )
    print(df.describe())
