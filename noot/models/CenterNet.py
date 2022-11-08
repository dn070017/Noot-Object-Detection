from noot.utils.TensorUtils import TensorUtils
import tensorflow as tf

class CenterNet(tf.keras.Model):
  def __init__(self):
    super().__init__()

    self.backbone = tf.keras.applications.resnet50.ResNet50(
      include_top=False,
      weights='imagenet',
      input_tensor=None,
      input_shape=None,
      pooling=None,
    )
    
    placeholder = tf.ones((1, 512, 512, 3))
    self.R = int(tf.cast(
        tf.shape(placeholder) / tf.shape(self.backbone(placeholder)), tf.int32)[1])

    self.keypoints_network = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same'),
      tf.keras.layers.Conv2D(filters=1, activation='sigmoid', kernel_size=1, strides=1),
    ])
    self.offset_network = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same'),
      tf.keras.layers.Conv2D(filters=2, activation='sigmoid', kernel_size=1, strides=1),
    ])
    self.size_network = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same'),
      tf.keras.layers.Conv2D(filters=2, activation='sigmoid', kernel_size=1, strides=1),
    ])

  def call(self, inputs, training):
    if isinstance(inputs, dict):
      inputs = inputs['image']
    if len(inputs.shape) == 3:
      tf.expand_dims(inputs, axis=0)

    result = self.backbone(inputs, training=training)
    
    keypoint_result = self.keypoints_network(result, training=training)
    offset_result = self.offset_network(result, training=training)
    size_result = self.size_network(result, training=training)
    
    return tf.concat([keypoint_result, offset_result, size_result], axis=-1)

  def train_step(self, data, lambda_offset=1, lambda_size=0.2):
    with tf.GradientTape() as tape:
      outputs = self(data, training = True)
      loss, keypoint_loss, offset_loss, size_loss = self.compute_loss(data, outputs, lambda_offset=lambda_offset, lambda_size=lambda_size)
    
    gradients = tape.gradient(loss, self.trainable_variables)
    gradients = TensorUtils.remove_nan_gradients(gradients)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return { 'loss': loss.numpy(), 'keypoint_loss': keypoint_loss.numpy(), 'offset_loss': offset_loss.numpy(), 'size_loss': size_loss.numpy() }

  def compute_loss(self, inputs, outputs, alpha=2, beta=4, lambda_offset=1, lambda_size=0.2):
    loss = 0
  
    label = inputs['objects/label']
    max_n_keypoints = label.shape[-1]
    n_keypoints = tf.reduce_sum(tf.where(label >= 0, tf.ones_like(label), tf.zeros_like(label)))
    n_keypoints = tf.cast(n_keypoints, tf.float32)

    """Keypoint"""
    n, height, width, channel = inputs['image'].shape
    n, lowres_height, lowres_width, n_map = outputs.shape

    bbox = inputs['objects/bbox']
    p_perc_x = (bbox[:, :, 3] + bbox[:, :, 1]) / 2
    p_perc_y = (bbox[:, :, 2] + bbox[:, :, 0]) / 2
    p_x = tf.round(p_perc_x * (width - 1))
    p_y = tf.round(p_perc_y * (height - 1))
    p_hat_x = tf.round(p_perc_x * (lowres_width - 1))
    p_hat_y = tf.round(p_perc_y * (lowres_height - 1))

    p_hat_x_tensor = tf.reshape(p_hat_x, (n, max_n_keypoints, 1, 1))
    p_hat_y_tensor = tf.reshape(p_hat_y, (n, max_n_keypoints, 1, 1))

    area = inputs['objects/area'] / (height * width) * (lowres_width * lowres_height)
    area = tf.cast(area, tf.float32)
    area = tf.where(area < 0, tf.zeros_like(area), area)

    pi = 3.1415926
    radius = tf.math.sqrt(0.3 * area / pi)
    p_hat_sigma = tf.reshape(
        2 / (3 * radius * tf.ones((n, max_n_keypoints)) + 1e-3), (n, max_n_keypoints, 1, 1))

    coord_x = tf.zeros((lowres_height, lowres_width)) + tf.cast(tf.range(lowres_width), tf.float32)
    coord_y = tf.transpose(tf.zeros((lowres_width, lowres_height)) + tf.cast(tf.range(lowres_height), tf.float32), [1, 0])
    coord_x = tf.expand_dims(coord_x, axis=0)
    coord_y = tf.expand_dims(coord_y, axis=0)
    coord_x = tf.repeat(coord_x, n, axis=0)
    coord_y = tf.repeat(coord_y, n, axis=0)
    coord_x = tf.reshape(coord_x, (n, 1, lowres_height, lowres_width))
    coord_y = tf.reshape(coord_y, (n, 1, lowres_height, lowres_width))

    y_pred = outputs[:, :, :, 0]
    y_true = tf.math.exp(-1 * ((coord_x-p_hat_x_tensor)** 2 + (coord_y-p_hat_y_tensor)**2) / p_hat_sigma)
    y_true = tf.reduce_max(y_true, axis=1)

    keypoint_loss = -1 / n_keypoints * tf.reduce_sum(
        tf.where(
            y_true == 1,
            (1 - y_pred) ** alpha * tf.math.log(y_pred + 1e-7),
            (1 - y_true) ** beta * y_pred ** alpha * tf.math.log(1 - y_pred + 1e-7)))
    loss += keypoint_loss

    """Offset"""
    sample_indices = tf.repeat(tf.reshape(tf.range(n), (-1, 1)), max_n_keypoints, axis=-1)
    offset_x = tf.squeeze(p_x - p_hat_x * self.R) / width
    offset_y = tf.squeeze(p_y - p_hat_y * self.R) / height

    offset_pred_x = outputs[:, :, :, 1]
    offset_pred_y = outputs[:, :, :, 2]
    indices = tf.concat([
      tf.expand_dims(sample_indices, axis=-1),
      tf.cast(tf.expand_dims(p_hat_x, axis=-1), tf.int32), 
      tf.cast(tf.expand_dims(p_hat_y, axis=-1), tf.int32)
    ], axis=-1)
    offset_pred_x = tf.gather_nd(offset_pred_x, indices)
    offset_pred_y = tf.gather_nd(offset_pred_y, indices)

    offset_pred_x = tf.reshape(offset_pred_x, (n, max_n_keypoints))
    offset_pred_y = tf.reshape(offset_pred_y, (n, max_n_keypoints))

    offset_loss = lambda_offset / n_keypoints * tf.reduce_sum(tf.where(
      inputs['objects/label'] < 0,
      tf.zeros_like(offset_pred_x),
      tf.math.sqrt((offset_x - offset_pred_x) ** 2 + (offset_y - offset_pred_y) ** 2)))
    loss += offset_loss

    """Size"""
    size_w = (bbox[:, :, 3] - bbox[:, :, 1])
    size_h = (bbox[:, :, 2] - bbox[:, :, 0])
    size_scaled_pred_w = outputs[:, :, :, 3]
    size_scaled_pred_h = outputs[:, :, :, 4]

    size_scaled_pred_w = tf.gather_nd(size_scaled_pred_w, indices)
    size_scaled_pred_h = tf.gather_nd(size_scaled_pred_h, indices)

    size_loss = lambda_size / n_keypoints * tf.reduce_sum(tf.where(
      inputs['objects/label'] < 0,
      tf.zeros_like(size_scaled_pred_w),
      tf.math.sqrt((size_w - size_scaled_pred_w) ** 2 + (size_h - size_scaled_pred_h) ** 2)))
    loss += size_loss
    
    return loss, keypoint_loss, offset_loss, size_loss
