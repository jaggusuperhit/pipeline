import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  # Ensures tokenization resources are available

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_pre-processing')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text: str) -> str:
    """
    Transform text by lowercasing, tokenizing, removing non-alphanumeric characters,
    removing stopwords, stemming, and joining back into a string.
    
    Args:
        text (str): Input text to transform
        
    Returns:
        str: Transformed text
        
    Raises:
        Exception: If text transformation fails
    """
    try:
        ps = PorterStemmer()
        text = text.lower()
        text = nltk.word_tokenize(text)
        text = [word for word in text if word.isalnum()]
        text = [word for word in text if word not in stopwords.words('english')]
        text = [ps.stem(word) for word in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error transforming text: {str(e)}")
        raise

def preprocess_df(df: pd.DataFrame, text_column: str = 'text', target_column: str = 'target', 
                 encoder: LabelEncoder = None, fit_encoder: bool = False) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Preprocess a DataFrame by transforming text and optionally encoding the target column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the text column to process
        target_column (str): Name of the target column to encode
        encoder (LabelEncoder, optional): Pre-fitted LabelEncoder
        fit_encoder (bool): Whether to fit the encoder (True) or transform using an existing one (False)
        
    Returns:
        tuple: Processed DataFrame and the LabelEncoder used
        
    Raises:
        Exception: If preprocessing fails
    """
    try:
        logger.debug('Starting DataFrame preprocessing')
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Drop rows with missing text
        original_len = len(df)
        df = df.dropna(subset=[text_column])
        dropped = original_len - len(df)
        if dropped > 0:
            logger.info(f'Dropped {dropped} rows with missing text')
        
        # Handle label encoding if target column exists
        if target_column in df.columns:
            if fit_encoder:
                if encoder is None:
                    encoder = LabelEncoder()
                df[target_column] = encoder.fit_transform(df[target_column])
            else:
                df[target_column] = encoder.transform(df[target_column])
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        logger.debug('Duplicates removed')
        
        # Transform text
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        
        return df, encoder
    
    except Exception as e:
        logger.error(f'Error during preprocessing: {str(e)}')
        raise

def main(text_column: str = 'text', target_column: str = 'target') -> None:
    """
    Main function to load, preprocess, and save train and test data.
    
    Args:
        text_column (str): Name of the text column
        target_column (str): Name of the target column
        
    Raises:
        Exception: If the main process fails
    """
    try:
        # Load data
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
        logger.info('Data loaded successfully')

        # Drop rows with missing target in train data
        if target_column in train_data.columns:
            original_len = len(train_data)
            train_data = train_data.dropna(subset=[target_column])
            dropped = original_len - len(train_data)
            if dropped > 0:
                logger.info(f'Dropped {dropped} rows with missing target in train data')

        # Preprocess train data and fit encoder
        train_processed, encoder = preprocess_df(
            train_data, text_column, target_column, fit_encoder=True
        )
        
        # Preprocess test data using the same encoder
        test_processed, _ = preprocess_df(
            test_data, text_column, target_column, encoder=encoder
        )

        # Save processed data
        data_path = os.path.join("data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.info(f'Processed data saved to {data_path}')
        
    except Exception as e:
        logger.error(f'Failed in main process: {str(e)}')
        raise

if __name__ == '__main__':
    main()