from flask import Flask
from threading import Thread
from flask import request
from PIL import Image
from io import BytesIO
import typing
import numpy as np
import cv2
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder


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
  image.save('image.jpg')
  configs_model_path = '../Models/03_handwriting_recognition/202301111911'
  configs_vocab = 'z9k5ijq.E0TPr,LcfDyumotYKO-QJ;d:Bnb8lNWHI4s6g7U!1A3)pweV#MRF"GZvax&h(S2C'

  model = ImageToWordModel(model_path=configs_model_path, char_list=configs_vocab)
  image_path = 'image.jpeg'
  image = cv2.imread(image_path)
  prediction_text = model.predict(image)
  return prediction_text

def run():
  app.run(port=8080)

def keep_alive():
  t = Thread(target=run)
  t.start()

keep_alive()
