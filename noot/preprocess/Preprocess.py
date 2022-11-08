from typing import List

import tensorflow as tf

class Preprocess:

  @staticmethod
  def flatten_dictionary(X, field: str = 'objects'):
    target =  X[field]
    for name, tensor in target.items():
      X[f"{field}/{name}"] = tensor
    del X[field]
    return X

  @staticmethod
  def normalize_image(X, field: str = 'image', digit: int = 255):
    X[field] /= digit
    return X

  @staticmethod
  def exclude_nonperson_label(
      X,
      white_list: List[str] = ['image', 'image/filename', 'image/id'],
      label_field: str = 'objects/label',
      crowd_field: str = 'objects/is_crowd',
      person_id: int = 0):
    labels = X[label_field]
    is_person = labels == person_id
    is_crowd = X.get(crowd_field, labels != person_id)
    not_crowd = tf.math.logical_not(is_crowd)
    person_indices = tf.where(is_person & not_crowd)[:, 0]
 
    for name, tensor in X.items():
      if name not in white_list:
        X[name] = tf.gather(tensor, person_indices, axis=0)
    
    return X

  @staticmethod
  def filter_nonperson(
      X,
      label_field: str = 'objects/label',
      crowd_field: str = 'objects/is_crowd',
      person_id: int = 0):
    labels = X[label_field]
    is_crowd = X.get(crowd_field, labels != person_id)
    condition = (labels == person_id) & tf.math.logical_not(is_crowd)
    return tf.reduce_any(condition)

  @staticmethod
  def rescale_image(
    X,
    max_n_objects: int = 5,
    scaled_width: int = 512,
    scaled_height: int = 512,
  ):
    scaled_image = tf.image.resize(X['image'], (scaled_height, scaled_width))
    if 'objects/label' in X:
      n_objects = len(X['objects/label'])
      n_padding = max_n_objects - n_objects
      if n_padding >= 0:
        area = tf.map_fn(fn=lambda x: scaled_width * (x[3] - x[1]) * scaled_height * (x[2] - x[0]), elems=X['objects/bbox'])
        area = tf.pad(area, tf.convert_to_tensor([[0, n_padding]]), 'CONSTANT', constant_values=-1.0)
        label = tf.pad(X['objects/label'], tf.convert_to_tensor([[0, n_padding]]), 'CONSTANT', constant_values=-1)
        bbox = tf.pad(X['objects/bbox'], tf.convert_to_tensor([[0, n_padding], [0, 0]]), 'CONSTANT', constant_values=-1.0)
      else:
        area = tf.cast(X['objects/area'][0:max_n_objects], tf.float32)
        label = X['objects/label'][0:max_n_objects]
        bbox = X['objects/bbox'][0:max_n_objects]

    X['image'] = scaled_image
    if 'objects/label' in X:
      X['objects/area'] = area
      X['objects/label'] = label
      X['objects/bbox'] = bbox
      del X['image/filename']
      del X['image/id']
      del X['objects/id']
      del X['objects/is_crowd']
  
    return X