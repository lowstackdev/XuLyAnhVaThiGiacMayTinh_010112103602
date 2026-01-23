from tensorflow.keras import regularizers
from tensorflow.keras.applications import InceptionV3, EfficientNetB3
from tensorflow.keras.layers import (
    Input,
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
    Average
)
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Recall
from tensorflow.keras import backend as K

INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 4

class RetinalFundusDiseaseClassifier(object):
    @staticmethod
    def _disease_classify_network(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, final_activation='softmax'):
        inputs = Input(shape=input_shape)
        # First conv block
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Second conv block
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Third conv block
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Flatten and dense layers
        x = Flatten()(x)
        x = Dense(1024, kernel_regularizer=regularizers.l2(0.016),
                  activity_regularizer=regularizers.l1(0.006),
                  bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
        x = Dropout(rate=0.45, seed=2022)(x)
        # classifier
        output = Dense(num_classes, activation=final_activation)(x)
        model = Model(inputs=inputs, outputs=output)
        return model

    @staticmethod
    def build_ensemble_average(models, input_shape=INPUT_SHAPE):
        input = Input(shape=input_shape)
        outputs = [model(input) for model in models]
        ensemble_output = Average()(outputs)
        ensemble_model = Model(inputs=input, outputs=ensemble_output)
        return ensemble_model
