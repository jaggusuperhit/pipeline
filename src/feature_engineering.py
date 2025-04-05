import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from typing import Tuple
import yaml

# Configuration
CONFIG = {
    'input_paths': {
        'train': os.path.join('data', 'interim', 'train_processed.csv'),
        'test': os.path.join('data', 'interim', 'test_processed.csv')
    },
    'output_paths': {
        'train': os.path.join('data', 'processed', 'train_tfidf.csv'),
        'test': os.path.join('data', 'processed', 'test_tfidf.csv')
    }
}

def setup_logging() -> logging.Logger:
    """Configure and return logger with file and console handlers."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('feature_engineering')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    log_file_path = os.path.join(log_dir, 'feature_engineering.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

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

logger = setup_logging()

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> None:
    """Validate that dataframe contains required columns."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        error_msg = f"Dataframe missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate data from a csv file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        
        validate_dataframe(df, required_columns=['text', 'target'])
        
        logger.info(f"Successfully loaded data from {file_path}")
        logger.debug(f"Data shape: {df.shape}, Columns: {df.columns.tolist()}")
        logger.debug(f"Label distribution:\n{df['target'].value_counts()}")
        
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply TF-IDF vectorization to the data."""
    try:
        logger.info(f"Applying TF-IDF with max_features={max_features}")
        
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        logger.info("Fitting TF-IDF vectorizer on training data...")
        X_train_bow = vectorizer.fit_transform(X_train)
        logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        logger.info("Transforming test data...")
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray(), 
                              columns=[f"feature_{i}" for i in range(max_features)])
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray(),
                             columns=[f"feature_{i}" for i in range(max_features)])
        test_df['label'] = y_test

        logger.info("TF-IDF transformation completed successfully")
        logger.debug(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error during TF-IDF transformation: {str(e)}")
        raise
        
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Data successfully saved to {file_path}")
        logger.debug(f"Saved data shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {str(e)}")
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']
        
        logger.info("Starting feature engineering process")
        
        # Load data
        logger.info("Loading training and test data")
        train_data = load_data(CONFIG['input_paths']['train'])
        test_data = load_data(CONFIG['input_paths']['test'])
        
        # Apply TF-IDF
        train_df, test_df = apply_tfidf(
            train_data, 
            test_data, 
            max_features
        )
        
        # Save results
        logger.info("Saving processed data")
        save_data(train_df, CONFIG['output_paths']['train'])
        save_data(test_df, CONFIG['output_paths']['test'])
        
        logger.info("Feature engineering completed successfully")
    except Exception as e:
        logger.critical(f"Feature engineering pipeline failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()