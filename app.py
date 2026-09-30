import os
import sys
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Resolve absolute paths relative to the current file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Global variables for model artifacts
model = None
vectorizer = None
load_error = None

def init_artifacts():
    """Safely load pickle files without throwing uncaught exceptions on startup."""
    global model, vectorizer, load_error
    
    if not os.path.exists(MODEL_PATH):
        load_error = f"model.pkl not found at {MODEL_PATH}"
        return

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
        if os.path.exists(VECTORIZER_PATH):
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
    except Exception as e:
        load_error = f"Failed to load pickle files: {str(e)}"

# Initialize models upon module import
init_artifacts()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', server_error=load_error)


@app.route('/predict', methods=['POST'])
def predict():
    # If the model failed to load during startup, gracefully alert the user
    if load_error or model is None:
        return render_template(
            'index.html', 
            error=f"Model Loading Error: {load_error or 'Model object is None.'}"
        )

    try:
        user_input = request.form.get('text_input', '').strip()

        if not user_input:
            return render_template('index.html', error="Please enter text to analyze.")

        # Transform or format input features
        if vectorizer is not None:
            features = vectorizer.transform([user_input])
        else:
            features = [user_input]

        # Model Inference
        prediction = model.predict(features)[0]

        # Calculate confidence score if model supports predict_proba
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(features)[0]
                confidence = round(float(max(probs)) * 100, 2)
            except Exception:
                confidence = None

        return render_template(
            'index.html', 
            prediction=prediction, 
            confidence=confidence, 
            user_input=user_input
        )

    except Exception as e:
        # Prevent runtime prediction errors from triggering a Vercel 500 crash
        return render_template(
            'index.html', 
            error=f"Prediction Runtime Error: {str(e)}"
        )


# Explicitly expose app for Vercel WSGI environment
app_handler = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
