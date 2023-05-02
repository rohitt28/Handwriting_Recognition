from flask import Flask
from threading import Thread
from flask import request
from PIL import Image
from io import BytesIO
import typing
import numpy as np
import cv2
import os
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text
      
app = Flask('')

@app.route('/')
def home():
  return "Server Alive!"


@app.route('/predict', methods=['POST'])
def predict():
  image_bytes = request.data
  image = Image.open(BytesIO(image_bytes))
  configs = BaseModelConfigs.load("Models/configs.yaml")
  image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
  model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
  prediction_text = model.predict(image)
  print(prediction_text)
  return prediction_text

app.run(host="0.0.0.0", port=3000)
