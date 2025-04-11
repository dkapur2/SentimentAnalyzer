# ✨ Sentence Sentiment Analysis

This project performs sentiment analysis on sentences using an LSTM-based neural network built with TensorFlow and Keras.

## 📊 Dataset

- Source: [Kaggle - Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- Columns used:
  - `text`: The tweet content
  - `airline_sentiment`: The label (positive or negative)

## ⚙️ Features

- Text preprocessing (cleaning, tokenizing, padding)
- Binary classification (positive vs. negative)
- LSTM model built with Keras
- Model training, evaluation, and visualization of accuracy/loss

## 🛠️ Getting Started

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   Open `Cleaned_Sentiment_Analysis.ipynb` in your Jupyter environment.

## 📁 Project Structure

```
sentiment-analysis/
├── Cleaned_Sentiment_Analysis.ipynb  # Annotated notebook
├── requirements.txt                  # Python package dependencies
├── data/                             # (Optional) dataset folder
```

## 🚀 Future Improvements

- Add a web interface (Streamlit or Gradio)
- Deploy as an API using Flask or FastAPI
- Host online (Render, Hugging Face Spaces)

## 📜 License

MIT License
