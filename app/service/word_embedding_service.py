import datetime

import gensim
import nltk
import numpy as np
import pandas as pd
from loguru import logger
from gensim.models import Word2Vec
from nltk.corpus import stopwords

from app.vo.model_information import ModelInfoVo

nltk.download("punkt")
nltk.download("stopwords")

VECTOR_SIZE = 150


class WordEmbeddingService:
    def train_model(self, text_df):
        logger.info("Word2Vec model training started.")
        tokenized_text = self.clean_and_tokenize_text(text_df)
        model = gensim.models.Word2Vec(
            tokenized_text, window=50, size=150, iter=5, min_count=3, workers=4
        )
        logger.info("Word2Vec model training completed.")
        model_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.save_model(model, model_id)
        model_info = ModelInfoVo()
        model_info.set_model_id(model_id + ".model")
        return model_info

    def create_word_embeddings(self, text_list, model_id):
        logger.info("Word Embeddings' generation is started.")
        model = self.get_model(model_id)
        vectors = [self.calculate_avg_vectors(x, model) for x in text_list]
        vectors_df = pd.DataFrame(vectors).apply(pd.Series).reset_index()
        logger.info("Word Embeddings' generation was completed.")
        return vectors_df

    def clean_and_tokenize_text(self, text_df):
        stop_words = stopwords.words("english")
        text_list = [
            [
                x.lower()
                for x in nltk.word_tokenize(x)
                if x not in stop_words and x.isalnum()
            ]
            for x in text_df["Text"]
        ]
        return text_list

    def save_model(self, model, model_id):
        model_path = "app/models/wv/" + model_id + ".model"
        model.save(model_path)
        logger.info("Word2Vec model was saved: %s", model_id)

    def get_model(self, model_id):
        model_path = "app/models/wv/" + model_id
        return Word2Vec.load(model_path)

    def calculate_avg_vectors(self, text_list, model):
        vectors = []
        for word in text_list:
            vectors.append(self.get_vector(word, model))
        if len(vectors) == 0:
            return np.zeros(VECTOR_SIZE)
        vec_avg = np.mean(vectors, axis=0)
        return vec_avg

    def get_vector(self, word, model):
        try:
            return model.wv.get_vector(word)
        except:
            return np.zeros(VECTOR_SIZE)
