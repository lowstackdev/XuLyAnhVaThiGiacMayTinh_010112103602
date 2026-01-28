import tensorflow as tf

from typing import List

def is_gpu_visibility() -> bool:
    """
    Checks whether the GPU is visible from TensorFlow.
    """

    recognized_gpu_devices: List = tf.config.list_physical_devices("GPU")
    print("List of recognized GPU devices:")
    print(recognized_gpu_devices)

    if not recognized_gpu_devices: return False

    # Sanity check - GPU is found, now test with simple matrix multiplication
    with tf.device('/gpu:0'):
        a = tf.constant(
            [1.0, 2.0, 3.0],
            shape=[1, 3],
            name='a'
        )
        b = tf.constant(
            [1.0, 2.0, 3.0],
            shape=[3, 1],
            name='b'
        )
        c = tf.matmul(a, b)

    print("Matrix multiplication result:")
    tf.print(c)

    return True