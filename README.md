# English to Italian Translator

A Streamlit web application for translating English text to Italian using a transformer-based neural network.

## Features

- Real-time translation from English to Italian
- Clean and intuitive user interface
- Uses beam search decoding for better translation quality

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. The app will open in your default web browser
3. Enter English text in the input area
4. Click "Translate to Italian" to get the translation

## Model Information

The application uses a transformer model trained on English-Italian translation data. The model files must be in the `opus_books_weights` directory. 