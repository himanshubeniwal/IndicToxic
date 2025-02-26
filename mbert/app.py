from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import json
import os

# Set the path to React build folder
REACT_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../my-react/build'))

# Initialize Flask with React build as static folder
app = Flask(__name__, static_folder=REACT_BUILD_DIR, static_url_path='')
CORS(app)  # Enable CORS for React

# Define Model Classes (Same as before)
class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name):
        super(EmbeddingModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

class ToxicityClassifier(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(ToxicityClassifier, self).__init__()
        self.fc = torch.nn.Linear(embedding_dim, 2)

    def forward(self, x):
        return self.fc(x)

class ModelTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.load_models()

    def load_models(self):
        self.languages = ["hi", "te"]
        for lang in self.languages:
            model_path = Path(f"models/{lang}/")
            config_path = model_path / "config.json"

            if not model_path.exists() or not config_path.exists():
                continue

            with open(config_path, "r") as f:
                config = json.load(f)

            if not (model_path / "embedding_model.pt").exists() or not (model_path / "classifier.pt").exists():
                continue

            tokenizer_path = model_path / "tokenizer"
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
            except Exception as e:
                continue

            embedding_model = EmbeddingModel(config["model_name"]).to(self.device)
            classifier = ToxicityClassifier(config["embedding_dim"]).to(self.device)

            embedding_model.load_state_dict(torch.load(model_path / "embedding_model.pt", map_location=self.device))
            classifier.load_state_dict(torch.load(model_path / "classifier.pt", map_location=self.device))

            embedding_model.eval()
            classifier.eval()

            self.models[lang] = {
                "tokenizer": tokenizer,
                "embedding": embedding_model,
                "classifier": classifier,
                "config": config
            }

    def predict(self, text, lang):
        if lang not in self.models:
            return jsonify({"error": f"Model for language '{lang}' not found"}), 400

        model_data = self.models[lang]
        tokenizer, embedding_model, classifier, config = (
            model_data["tokenizer"],
            model_data["embedding"],
            model_data["classifier"],
            model_data["config"]
        )

        encoding = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config["max_length"],
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            embeddings = embedding_model(input_ids, attention_mask)
            outputs = classifier(embeddings)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)

        return {
            "toxicity": round(probabilities[0][1].item() * 100, 2),
            "is_toxic": bool(prediction.item())
        }


# Initialize Model Tester
tester = ModelTester()

# API Route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "").strip()
    lang = data.get("lang", "hi")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = tester.predict(text, lang)
    return jsonify(result)

# Serve React Frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(REACT_BUILD_DIR, path)):
        return send_from_directory(REACT_BUILD_DIR, path)
    else:
        return send_from_directory(REACT_BUILD_DIR, 'index.html')

if __name__ == '__main__':
    app.run(host='10.0.62.187', port=3339)