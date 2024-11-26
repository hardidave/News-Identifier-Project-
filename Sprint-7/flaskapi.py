from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re
from textblob import TextBlob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
try:
    MODEL_PATH = r'C:\Users\patel\Downloads\fake_news_detector_20241124_221416.joblib'
    loaded_data = joblib.load(MODEL_PATH)
    model = loaded_data['model']
    vectorizer = loaded_data['vectorizer']
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    vectorizer = None

def preprocess_text(text):
    """Basic text preprocessing"""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_sentiment(text):
    """Simple sentiment analysis"""
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

@app.route('/health', methods=['GET'], )
def health_check():
    """Simple health check"""
    return jsonify({
        'status': 'healthy' if model and vectorizer else 'error',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text
        data = request.get_json()
        print(request.get_data())
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        if not text or len(text.strip()) < 10:
            return jsonify({'error': 'Text too short'}), 400

        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Get prediction
       # Get prediction
        text_vectorized = vectorizer.transform([processed_text])
        prediction = int(model.predict(text_vectorized)[0])  # Convert to Python int
        probabilities = model.predict_proba(text_vectorized)[0].tolist()  # Convert to Python list
        sentiment = analyze_sentiment(text)
        # Prepare response
        response = {
            'prediction': {
                'label': 'Real News' if prediction == 1 else 'Fake News',
                'is_fake': bool(prediction == 0),  # Convert to Python bool
                'confidence': float(max(probabilities))  # Convert to Python float
            },
            'probability': {
                'fake': float(probabilities[0]),  # Convert to Python float
                'real': float(probabilities[1])   # Convert to Python float
            },
            'sentiment': {
                'polarity': float(sentiment['polarity']),  # Convert to Python float
                'subjectivity': float(sentiment['subjectivity'])  # Convert to Python float
            },
            'text_stats': {
                'length': len(text),
                'word_count': len(text.split())
            }
        }

        return jsonify(response)


    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not model or not vectorizer:
        logger.error("Model not loaded")
        exit(1)

    print("\nStarting Fake News Detection API...")
    print("Server running at: http://127.0.0.1:5000")
    print("Press CTRL+C to quit\n")
    
    app.run(host='127.0.0.1', port=5000)