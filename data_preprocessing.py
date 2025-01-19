import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import subprocess


def load_and_preprocess_data():
    
        # Create the ~/.kaggle directory if it doesn't exist
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Copy the kaggle.json file to the ~/.kaggle directory
    kaggle_json_source = "kaggle.json"  # Update this if your file is elsewhere
    kaggle_json_destination = os.path.join(kaggle_dir, "kaggle.json")
    
    try:
        # Copy the kaggle.json file
        subprocess.run(["cp", kaggle_json_source, kaggle_json_destination], check=True)
        print(f"Copied kaggle.json to {kaggle_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error copying kaggle.json: {e}")
        return
    
    # Set the correct permissions for kaggle.json
    try:
        subprocess.run(["chmod", "600", kaggle_json_destination], check=True)
        print("Set correct permissions for kaggle.json")
    except subprocess.CalledProcessError as e:
        print(f"Error setting permissions: {e}")
        return
    
    # Download the dataset
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", "zynicide/wine-reviews"], check=True)
        print("Dataset downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")

     # Load dataset
     
    zip_path = 'wine-reviews.zip'
    data_path = 'files'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    csv_path = os.path.join(data_path, 'winemag-data-130k-v2.csv')
    df = pd.read_csv(csv_path, index_col=0).dropna()

    # Text preprocessing
    X_text = df['description'].values
    y = df['price'].values
    tk = Tokenizer()
    tk.fit_on_texts([text_to_word_sequence(sentence) for sentence in X_text])
    X_tokens = tk.texts_to_sequences(X_text)
    vocab_size = len(tk.word_index)
    maxlen = 60
    X_pad = pad_sequences(X_tokens, dtype=float, padding='post', maxlen=maxlen)

    # Numerical preprocessing
    ohe = OneHotEncoder(sparse_output=False)
    X_region_ohe = ohe.fit_transform(df[['region_1', 'region_2', 'variety']])
    X_num = np.hstack([df[['points']].values, X_region_ohe])

    #Split data into test and train

    X_pad_train, X_pad_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_pad, X_num, y, test_size=0.2, random_state=42
)

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_num_train)
    X_num_test = scaler.transform(X_num_test)

    return X_pad_train, X_pad_test, X_num_train, X_num_test, y_train, y_test, vocab_size, maxlen
