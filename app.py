import os
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Resolve absolute paths relative to the current file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

def load_pkl(path):
    """Safely load pickle files if they exist."""
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file at {path}: {str(e)}")
            return None
    return None

# Load model and vectorizer at app initialization
model = load_pkl(MODEL_PATH)
vectorizer = load_pkl(VECTORIZER_PATH)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Diagnostic check: ensure model is loaded
    if model is None:
        return render_template(
            'index.html', 
            error="Server Error: Model file (model.pkl) failed to load or is missing on Vercel."
        )

    try:
        user_text = request.form.get('text_input', '').strip()

        if not user_text:
            return render_template('index.html', error="Please enter text to analyze.")

        # Vectorization/Preprocessing step
        if vectorizer is not None:
            features = vectorizer.transform([user_text])
        else:
            features = [user_text]

        # Model Inference
        prediction = model.predict(features)[0]

        # Calculate confidence score if supported by the model
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
            user_input=user_text
        )

    except Exception as e:
        # Catch runtime inference errors gracefully without crashing the server
        return render_template(
            'index.html', 
            error=f"Prediction Error: {str(e)}"
        )

# Required for local testing; Vercel imports 'app' automatically
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
