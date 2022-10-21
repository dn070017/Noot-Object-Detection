from noot.models.CenterNetLite import CenterNetLite
from noot.preprocess.Preprocess import Preprocess
from time import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import seaborn as sns

try:
  physical_devices = tf.config.list_physical_devices('GPU')
  for i, d in enumerate(physical_devices):
      tf.config.experimental.set_memory_growth(physical_devices[i], True)
except:
  print('No GPU detected. Use CPU instead')

dataset = tfds.load('coco')
try:
  dataset = tf.data.experimental.load('datasets/coco')
except:
  dataset = dataset['train'].map(Preprocess.flatten_dictionary).filter(Preprocess.filter_nonperson).map(Preprocess.exclude_nonperson_label).map(Preprocess.normalize_image).map(Preprocess.rescale_image)
  tf.data.experimental.save(dataset, 'datasets/coco')

model = CenterNetLite()
model.compile(run_eagerly=True)
model.optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4)

batch_size = 16
val_images = 1024
max_n_epochs = 256
best_loss = 10000
patience = 32
current_patience = 32
for epoch in range(max_n_epochs):
  start_time = time()
  train_loss_list = list()
  val_loss_list = list()
  
  for j, batch in enumerate(dataset.skip(val_images).batch(batch_size)):
    losses = model.train_step(batch, lambda_offset=4, lambda_size=10)
    train_loss_list.append((losses['loss'], losses['keypoint_loss'], losses['offset_loss'],losses['size_loss']))
  for batch in dataset.take(val_images).batch(batch_size):
    outputs = model(batch, training=False)
    loss, keypoint_loss, offset_loss, size_loss = model.compute_loss(batch, outputs, lambda_offset=4, lambda_size=10)
    val_loss_list.append((loss, keypoint_loss, offset_loss, size_loss))

  train_loss = np.array(train_loss_list)
  val_loss = np.array(val_loss_list)
  mean_train_loss = np.mean(train_loss, axis=0)
  mean_val_loss = np.mean(val_loss, axis=0)

  if best_loss > mean_val_loss[0]:
    best_loss = mean_val_loss[0]
    model.save_weights(f'logging/centernet')
    current_patience = patience
  else:
    current_patience -= 1
  print(f"Epoch {epoch+1:>03}: {mean_train_loss[0]:>10.3f}{mean_train_loss[1]:>10.3f}{mean_train_loss[2]:>10.3f}{mean_train_loss[3]:>10.3f}\t({int(time() - start_time)})")
  print(f"         : {mean_val_loss[0]:>10.3f}{mean_val_loss[1]:>10.3f}{mean_val_loss[2]:>10.3f}{mean_val_loss[3]:>10.3f}\t({patience})")
  if current_patience == 0:
    break

model.load_weights(f'logging/centernet')
