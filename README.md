# Wine Price Prediction Using NLP and Numerical Features 🍷

This project leverages a comprehensive dataset of over 150,000 wine reviews from WineEnthusiast to predict wine prices. By utilizing Natural Language Processing (NLP) and dense neural network layers, this model analyzes both the textual content of wine reviews and numerical features to estimate the price of wines accurately.

# Dataset

The dataset includes:

- winemag-data-130k-v2.csv containing 10 columns and 130k rows of wine reviews.
- winemag-data_first150k.csv containing 10 columns and 150k rows of wine reviews.
- winemag-data-130k-v2.json containing 6919 nodes of wine reviews.

# Models

### Numerical Model
The numerical model is implemented using dense neural network layers and processes numerical features such as regions, variety and points.

### NLP Model
The NLP model processes the textual content of wine reviews using embedding layers, convolutional layers, and dense layers.

# Training
The model is trained using a combined dataset of textual and numerical features. Early stopping is employed to prevent overfitting.

# Results

The training process includes monitoring loss and Mean Absolute Error (MAE) to evaluate the model's performance. 

![image](https://github.com/user-attachments/assets/b631a8b7-b883-46cd-9785-30948887f933)

# Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

# Installation 

```bash
git clone git@github.com:nicole-baltodano/Wine_price_prediction_from_review.git
cd Wine_price_prediction_from_review
pip install -r requirements.txt
```

# Acknowledgements

The data was downloaded from Kaggle Wine Reviews Dataset.
https://www.kaggle.com/datasets/zynicide/wine-reviews


