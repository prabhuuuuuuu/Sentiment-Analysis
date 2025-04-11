# Movie Review Sentiment Analysis

A deep learning model built with TensorFlow to classify movie reviews as positive or negative based on the IMDB dataset of 50K reviews.

## Project Overview
This project implements a sentiment analysis neural network that can determine whether a movie review expresses positive or negative sentiment. The model is trained on IMDB's dataset of 50,000 movie reviews, which is a widely used benchmark for binary sentiment classification tasks.

## Features
- Text preprocessing using TensorFlow's Tokenizer
- Word embedding representation
- Neural network with multiple dense layers and dropout for regularization
- Binary classification with sigmoid activation
- Performance evaluation on test dataset

## Technical Implementation
- **Framework**: TensorFlow with Keras API
- **Model Architecture**: Sequential model with:
  - Embedding layer to convert tokens to dense vectors
  - Global Average Pooling to reduce dimensionality
  - Multiple Dense layers with ReLU activation
  - Dropout layers to prevent overfitting
  - Output layer with sigmoid activation for binary classification
- **Preprocessing**: Tokenization and sequence padding
- **Training**: Adam optimizer with binary cross-entropy loss

## Dataset
The model uses the IMDB Dataset of 50K Movie Reviews, which contains an equal number of positive and negative reviews. The dataset is preprocessed by converting text to sequences and padding to ensure uniform input size.

## Results
The model achieves approximately 85-90% accuracy on the test set, demonstrating effective sentiment classification capabilities.

## Dependencies
- Python 3.x
- TensorFlow
- NumPy
- Pandas
- scikit-learn

## Usage
1. Install the required dependencies
2. Run the script to train and evaluate the model:
```python
python sentiment_analysis.py
```

## Future Improvements
- Experiment with bidirectional LSTM or transformer architectures
- Implement attention mechanisms
- Try different preprocessing techniques
- Expand to multi-class sentiment analysis
