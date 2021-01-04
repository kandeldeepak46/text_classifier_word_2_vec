import json
from loguru import logger
import pandas as pd
from flask import Blueprint, current_app, request, jsonify

from app.service.prediction_service import PredictionService
from app.service.text_classifier_service import TextClassifierService
from app.service.word_embedding_service import WordEmbeddingService

controller = Blueprint("classifier-controller", __name__)

wordEmbeddingService = WordEmbeddingService()
textClassifierService = TextClassifierService()
predictionService = PredictionService()

@controller.route('/healthz')
def test_health():
    return "<h1>The api is running smoothly</h1>"

@controller.route("/wv-model-training", methods=["POST"])
def train_wv_model():
    logger.info("Model training started.")
    csv_file = request.form["file"]
    with open(csv_file, encoding="utf8") as file:
        text_df = pd.read_csv(file)
    model_info = wordEmbeddingService.train_model(text_df)
    return jsonify(json.dumps(model_info.__dict__))


@controller.route("/text-classifier-training", methods=["POST"])
def train_classifier_model():
    logger.info("Text classifier model training started.")
    model_id = request.form["model_id"]
    csv_file = request.form["file"]
    with open(csv_file, encoding="utf8") as file:
        text_df = pd.read_csv(file)
    model_info = textClassifierService.train_classifier_model(text_df, model_id)
    return jsonify(json.dumps(model_info.__dict__))


@controller.route("/predict-text", methods=["POST"])
def predict_text_class():
    logger.info("Text prediction started.")
    prediction_request = request.get_json()
    text = prediction_request["text"]
    model_id = prediction_request["model_id"]
    result = predictionService.predict_text_class(text, model_id)
    return jsonify(json.dumps(result.__dict__))
