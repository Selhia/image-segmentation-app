

# Convertir un modèle Keras en TFLite
import tensorflow as tf
from model_architecture import create_unet_resnet50

# Chemins
h5_path = "model/unet_resnet50_cityscapes.h5"
tflite_path = "model/unet_resnet50_cityscapes.tflite"

# Paramètres
input_shape = (256, 512, 3)
n_classes = 8

# Charger le modèle
model = create_unet_resnet50(input_shape=input_shape, n_classes=n_classes)
model.load_weights(h5_path)

# Conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Sauvegarde
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Modèle converti en TFLite et enregistré : {tflite_path}")
