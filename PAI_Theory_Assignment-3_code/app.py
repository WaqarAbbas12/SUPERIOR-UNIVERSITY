from flask import Flask, request, jsonify
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
  text = text.lower()
  text = ''.join([char for char in text if char not in string.punctuation])
  text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
  lemmatizer = WordNetLemmatizer()
  text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
  return text


def load_model_and_vectorizer(model_path="spam_classifier_model.pkl", vectorizer_path="vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully")
    return model, vectorizer


# Classify a single input row
def classify_message(message, model, vectorizer):
    preprocessed_message = preprocess_text(message)
    transformed_message = vectorizer.transform([preprocessed_message])
    # print(transformed_message)x
    prediction = model.predict(transformed_message) # 0 or 1 Spam and 0 for Ham
    return "Spam" if prediction[0] == 1 else "Ham"



# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer at startup
loaded_model, loaded_vectorizer = load_model_and_vectorizer()

# Define API route for classifying messages
@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get input data (JSON format)
        data = request.get_json()
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "Message text is required"}), 400

        # Classify the message
        result = classify_message(message, loaded_model, loaded_vectorizer)

        # Return response
        return jsonify({"message": message, "classification": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)

