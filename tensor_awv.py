import numpy as np
import os

from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.text_classifier import AverageWordVecSpec
from tflite_model_maker.text_classifier import DataLoader

import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')

spec = model_spec.get('average_word_vec')

train_data = DataLoader.from_csv(
      filename='custom.csv',
      text_column='taskName',
      label_column='iconId',
      model_spec=spec,
      is_training=True)
      
test_data = DataLoader.from_csv(
      filename='test.csv',
      text_column='taskName',
      label_column='iconId',
      model_spec=spec,
      is_training=False)


model = text_classifier.create(train_data, model_spec=spec, epochs=2000)


loss, acc = model.evaluate(test_data)



model.export(export_dir='ic/', export_format=[ExportFormat.LABEL, ExportFormat.VOCAB,ExportFormat.TFLITE])


