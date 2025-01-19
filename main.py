
from data_preprocessing import load_and_preprocess_data
from model_builder import build_combined_model, train_model, plot_training_results, plot_scatter_actual_vs_predicted

def main():
    # Load and preprocess data
    X_pad_train, X_pad_test, X_num_train, X_num_test, y_train, y_test, vocab_size, maxlen = load_and_preprocess_data()

    # Build and train the combined model
    model_combined, history, predictions = train_model(X_pad_train, X_pad_test, X_num_train, X_num_test, y_train, vocab_size, maxlen)

    # Plot training results
    plot_training_results(history)

    # Plot scatter of actual vs. predicted prices
    plot_scatter_actual_vs_predicted(y_test, predictions)

if __name__ == "__main__":
    main()
