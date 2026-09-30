import pickle
import os
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load model and vectorizer/preprocessor securely
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

model = None
vectorizer = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model artifacts: {e}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded correctly.'}), 500

    try:
        data = request.form.get('text_input', '')
        
        if not data:
            return render_template('index.html', error="Please enter text to analyze.")

        # Transform input if vectorizer exists, otherwise format directly
        if vectorizer:
            transformed_input = vectorizer.transform([data])
            prediction = model.predict(transformed_input)[0]
            
            # Retrieve probabilities if supported
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(transformed_input)[0]
                confidence = round(max(probs) * 100, 2)
            else:
                confidence = None
        else:
            prediction = model.predict([data])[0]
            confidence = None

        return render_template(
            'index.html', 
            prediction=prediction, 
            confidence=confidence, 
            user_input=data
        )

    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}")

# Vercel needs the app object to be exposed directly
if __name__ == '__main__':
    app.run(debug=True)
