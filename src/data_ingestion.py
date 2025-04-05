import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import csv
import yaml

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# Clear any existing handlers to avoid duplicate logs
if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def inspect_csv_file(file_path, problematic_line=42):
    """Inspect a specific line in the CSV file to diagnose parsing issues."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.readlines()
            
        if len(lines) >= problematic_line:
            logger.debug(f"Problematic line {problematic_line}: {lines[problematic_line-1]}")
            # Print a few lines before and after for context
            start = max(0, problematic_line-3)
            end = min(len(lines), problematic_line+2)
            
            logger.debug(f"Context (lines {start+1} to {end}):")
            for i in range(start, end):
                logger.debug(f"Line {i+1}: {lines[i]}")
    except Exception as e:
        logger.error(f"Error inspecting file: {e}")

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file with special handling for parsing errors."""
    try:
        logger.debug(f'Attempting to load data from {data_path}')
        
        # First inspect the problematic line
        inspect_csv_file(data_path)
        
        # Try multiple approaches to load the CSV
        try:
            # Approach 1: Using on_bad_lines='skip' (for newer pandas versions)
            df = pd.read_csv(data_path, encoding='utf-8', on_bad_lines='skip')
        except TypeError:
            try:
                # Approach 2: Using error_bad_lines=False (for older pandas versions)
                df = pd.read_csv(data_path, encoding='utf-8', error_bad_lines=False)
            except:
                # Approach 3: Using more specific CSV options
                df = pd.read_csv(data_path, encoding='utf-8', quoting=csv.QUOTE_NONE, 
                                 escapechar='\\', delimiter=',', engine='python')
        
        logger.debug(f'Successfully loaded data with shape {df.shape}')
        
        # Check data integrity
        logger.debug(f'DataFrame columns: {df.columns.tolist()}')
        logger.debug(f'First few rows:\n{df.head()}')
        
        return df
    except Exception as e:
        logger.error(f'Failed to load data: {e}')
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # List all columns to check if they exist
        logger.debug(f"Available columns before preprocessing: {df.columns.tolist()}")
        
        # Check if unnamed columns exist before dropping
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)
            logger.debug(f"Dropped unnamed columns: {unnamed_cols}")
        
        # Check if target columns exist before renaming
        if 'v1' in df.columns and 'v2' in df.columns:
            df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
            logger.debug("Renamed columns 'v1' to 'target' and 'v2' to 'text'")
        else:
            logger.warning(f"Could not find 'v1' or 'v2' columns. Current columns: {df.columns.tolist()}")
        
        # Drop duplicates if any
        original_len = len(df)
        df.drop_duplicates(inplace=True)
        logger.debug(f"Removed {original_len - len(df)} duplicate rows")
        
        logger.debug(f"Data preprocessing completed. Final shape: {df.shape}")
        return df
    except KeyError as e:
        logger.error(f'Missing column in the dataframe: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error during preprocessing: {e}')
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        
        train_file = os.path.join(raw_data_path, "train.csv")
        test_file = os.path.join(raw_data_path, "test.csv")
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        logger.debug(f"Train data saved to {train_file} with shape {train_data.shape}")
        logger.debug(f"Test data saved to {test_file} with shape {test_data.shape}")
    except Exception as e:
        logger.error(f'Unexpected error occurred while saving the data: {e}')
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # Configuration parameters
        # test_size = 0.2
        random_state = 2
        
        # Try local file path first (more reliable)
        data_path = r"D:\ML-Ops\pipeline\experiments\spam.csv"
        
        # If local file doesn't exist, try downloading from GitHub
        if not os.path.exists(data_path):
            logger.warning(f"Local file {data_path} not found, trying GitHub URL")
            data_path = 'https://raw.githubusercontent.com/jaggusuperhit/pipeline/main/experiments/spam.csv'
        
        # Load the data
        logger.info(f"Starting data ingestion process using {data_path}")
        df = load_data(data_path=data_path)
        
        # Preprocess the data
        logger.info("Starting data preprocessing")
        final_df = preprocess_data(df)
        
        # Split into train and test sets
        logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=random_state
        )
        
        # Save the data
        logger.info("Saving train and test datasets")
        save_data(train_data, test_data, data_path='./data')
        
        logger.info("Data ingestion process completed successfully")
        print("Data ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f'Failed to complete the data ingestion process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()