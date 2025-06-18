import tensorflow as tf
# Assurez-vous que model_architecture.py est dans le même dossier ou accessible dans le PYTHONPATH
from model_architecture import create_unet_resnet50

# Chemins
h5_path = "model/unet_resnet50_cityscapes.h5"
tflite_path = "model/unet_resnet50_cityscapes.tflite"

# Paramètres du modèle (doivent correspondre à votre app.py)
input_shape = (256, 512, 3) # Hauteur, Largeur, Canaux
n_classes = 8

# 1. Charger le modèle Keras original
model = create_unet_resnet50(input_shape=input_shape, n_classes=n_classes)
model.load_weights(h5_path)

# --- CORRECTION ICI : DÉFINITION DE LA FONCTION CONCRÈTE POUR LA CONVERSION ---

# Définir une fonction TF qui encapsule l'appel du modèle Keras.
# L'input_signature est CRUCIALE ici pour que le convertisseur TFLite puisse
# inférer correctement la forme et le type des entrées.
@tf.function(input_signature=[
    # La forme est [batch_size, hauteur, largeur, canaux].
    # Utilisez 1 pour la taille du batch si vous traitez une image à la fois.
    tf.TensorSpec(shape=[1, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32, name='input_image')
])
def serving_fn(input_image):
    # Appeler directement le modèle Keras à l'intérieur de la tf.function
    return model(input_image)

# Obtenir la fonction concrète pour la conversion
concrete_func = serving_fn.get_concrete_function()

# 2. Créer le convertisseur TFLite à partir de la fonction concrète
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# Ne pas appliquer d'optimisation de quantification pour garder tout en FLOAT32
converter.optimizations = []


# 3. Convertir le modèle
try:
    tflite_model = converter.convert()

    # 4. Sauvegarder le modèle TFLite
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Modèle Keras converti en TFLite et sauvegardé sous : {tflite_path}")

except Exception as e:
    print(f"Erreur grave lors de la conversion du modèle TFLite : {e}")
    print("Vérifiez l'architecture de votre modèle Keras et la compatibilité avec le convertisseur TFLite.")
    print("Essayez éventuellement une autre version de TensorFlow pour la conversion si le problème persiste.")