import tensorflow as tf

from typing import List

class TensorUtils:
  """TensorUtils

  Class containing multiple utility functions for tf.Tensor.

  """

  @staticmethod
  def remove_nan_gradients(gradients: List[tf.Tensor], clip_value=0.1) -> List[tf.Tensor]:
    """Replace nan, inf with 0 and perform gradient clipping for the 
    gradients computed by tf.GradientTape.gradient().

    Parameters
    ----------
    gradients: List[tf.Tensor]
        gradients computed by tf.GradientTape.gradient()

    clip_value: float, optional
        clip values for gradient clipping. The resulting gradient will 
        be in the range of [-clip_value, clip_value]
    
    Returns
    -------
    List[tf.Tensor]
        processed gradients.
    
    """
    for i, g in enumerate(gradients):
      if gradients[i] is None:
        continue
      gradients[i] = tf.where(tf.math.is_nan(g), tf.zeros_like(g), g)
      gradients[i] = tf.where(
          tf.math.is_inf(gradients[i]),
          tf.zeros_like(gradients[i]),
          gradients[i])
      gradients[i] = tf.where(
          gradients[i] > clip_value,
          clip_value * tf.ones_like(gradients[i]),
          gradients[i])
      gradients[i] = tf.where(
          gradients[i] < -1 * clip_value,
          -1 * clip_value * tf.ones_like(gradients[i]),
          gradients[i])
          
    return gradients