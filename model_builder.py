from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def build_model_nlp(vocab_size, maxlen, embedding_size=40):
    return Sequential([
        layers.Embedding(input_dim=vocab_size + 1, input_length=maxlen, output_dim=embedding_size, mask_zero=True),
        layers.Conv1D(10, kernel_size=15, padding='same', activation="relu"),
        layers.Conv1D(10, kernel_size=10, padding='same', activation="relu"),
        layers.Flatten(),
        layers.Dense(30, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(1, activation='relu'),
    ])

def build_model_num(input_dim):
    input_num = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(input_num)
    x = layers.Dense(32, activation="relu")(x)
    output_num = layers.Dense(1, activation="relu")(x)
    return models.Model(inputs=input_num, outputs=output_num)

def build_combined_model(vocab_size, maxlen, input_dim):
    model_nlp = build_model_nlp(vocab_size, maxlen)
    model_num = build_model_num(input_dim)

    combined = layers.concatenate([model_nlp.output, model_num.output])
    x = layers.Dense(10, activation="relu")(combined)
    outputs = layers.Dense(1, activation="linear")(x)

    return models.Model(inputs=[model_nlp.input, model_num.input], outputs=outputs)

import numpy as np

def train_model(X_pad_train, X_pad_test, X_num_train, X_num_test, y_train, vocab_size, maxlen):
    input_dim = X_num_train.shape[1]
    model_combined = build_combined_model(vocab_size, maxlen, input_dim)

    model_combined.compile(loss="mse", optimizer=Adam(learning_rate=1e-4), metrics=['mae'])
    es = EarlyStopping(patience=2)

    history = model_combined.fit(
        x=[X_pad_train, X_num_train],
        y=y_train,
        validation_split=0.3,
        epochs=100,
        batch_size=32,
        callbacks=[es]
    )

    # Save the trained model
    model_combined.save("wine_price_model.h5")
    print("Model saved to wine_price_model.h5")

    # Generate predictions
    predictions = model_combined.predict([X_pad_test, X_num_test])

    return model_combined, history, predictions


def plot_training_results(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_mae, label='Training MAE')
    plt.plot(val_mae, label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.show()

def plot_scatter_actual_vs_predicted(y_actual, y_predicted):

    # Calculate MAE
    mae = mean_absolute_error(y_actual, y_predicted)

    plt.figure(figsize=(8, 6))

    # Scatter plot with improved transparency
    plt.scatter(y_actual, y_predicted, alpha=0.6, color="blue", edgecolor="k", label="Predictions")

    # Add 45° reference line
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()],
             color="red", linestyle="--", linewidth=2, label="Perfect Predictions")

    # Set limits to focus on majority of the data
    plt.xlim(0, 150)
    plt.ylim(0, 150)

    # Titles and labels
    plt.title("Actual vs Predicted Wine Prices", fontsize=14)
    plt.xlabel("Actual Prices ($)", fontsize=12)
    plt.ylabel("Predicted Prices ($)", fontsize=12)

    # Annotate MAE on the plot
    plt.text(10, 100, f"MAE = {mae:.2f}", fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.7))


    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()
