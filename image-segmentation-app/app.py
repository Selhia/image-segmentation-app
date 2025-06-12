import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from tensorflow.keras.preprocessing.image import img_to_array, load_img # Garder pour la préparation de l'image
from PIL import Image
from skimage.color import label2rgb
import tensorflow as tf # Importer TensorFlow


app = Flask(__name__)

# --- Configuration des dossiers ---
UPLOAD_FOLDER = 'static/images/uploads'
MASK_FOLDER = 'static/images/masks'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER

# Créer les dossiers s'ils n'existent pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# --- Paramètres du modèle ---
# Le INPUT_SHAPE reste le même, car votre modèle TFLite attend la même entrée
INPUT_SHAPE = (256, 512, 3) # Hauteur, Largeur, Canaux
NUM_CLASSES = 8            # Nombre de classes

# --- Chemin du modèle TFLite ---
MODEL_PATH_TFLITE = 'model/unet_resnet50_cityscapes.tflite' # NOUVEAU CHEMIN

# --- Palette de couleurs pour Cityscapes (comme avant) ---
CITYSCAPES_COLOR_PALETTE = [
    (0, 0, 0),        # 0: arrière-plan / void
    (128, 64, 128),   # 1: route
    (244, 35, 232),   # 2: trottoir
    (70, 70, 70),     # 3: bâtiment
    (102, 102, 156),  # 4: mur
    (190, 153, 153),  # 5: clôture
    (153, 153, 153),  # 6: poteau
    (250, 170, 30),   # 7: panneau de signalisation
    # ... (autres si nécessaire)
]

# --- Chargement du modèle TensorFlow Lite ---
interpreter = None
input_details = None
output_details = None

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH_TFLITE)
    interpreter.allocate_tensors() # Alloue de la mémoire pour les tenseurs

    # Récupérer les détails des tenseurs d'entrée et de sortie
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Modèle TensorFlow Lite '{MODEL_PATH_TFLITE}' chargé avec succès.")
    print(f"Détails de l'entrée TFLite : {input_details}")
    print(f"Détails de la sortie TFLite : {output_details}")

except Exception as e:
    print(f"Erreur lors du chargement du modèle TensorFlow Lite : {e}")
    print("Vérifiez la présence de 'unet_resnet50_cityscapes.tflite' et l'intégrité du fichier.")
    interpreter = None

# --- Fonction de segmentation mise à jour pour TFLite ---
def get_segmentation_mask(image_path, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1])):
    if interpreter is None:
        return None, "Erreur: Le modèle TFLite n'a pas pu être chargé. Veuillez vérifier le serveur."

    try:
        # Charger et prétraiter l'image (comme avant)
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Ajoute la dimension du batch

        # Assurer que le type de données de l'entrée correspond à ce que le modèle TFLite attend
        # Les modèles TFLite peuvent attendre des Float32 ou des UInt8 normalisés.
        # Vérifiez input_details[0]['dtype']
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.float32:
            img_array = img_array / 255.0 # Normalisation pour les modèles Float32
        elif input_dtype == np.uint8:
            # Si le modèle attend des UInt8, assurez-vous que les valeurs sont entre 0 et 255
            # et que le type est uint8. Pas de normalisation si c'est le cas.
            img_array = img_array.astype(np.uint8)

        # Copier le tenseur d'entrée
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Exécuter l'inférence
        interpreter.invoke()

        # Récupérer le tenseur de sortie
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Post-traitement (comme avant, car la sortie est similaire)
        # La sortie de TFLite sera [1, H, W, N_CLASSES], comme Keras
        segmentation_map = np.argmax(prediction[0], axis=-1)

        # Conversion du masque de segmentation en une image RGB colorée
        colored_mask = label2rgb(
            segmentation_map,
            colors=np.array(CITYSCAPES_COLOR_PALETTE)/255.0,
            bg_label=0,
            image=img_array[0].astype(np.float64) if input_dtype == np.float32 else img_array[0], # Passer l'image originale pour superposition (doit être float)
            alpha=0.5
        )
        colored_mask = (colored_mask * 255).astype(np.uint8)

        mask_image = Image.fromarray(colored_mask)

        return mask_image, None

    except Exception as e:
        return None, f"Erreur lors de la prédiction ou du traitement du masque TFLite : {e}"

# --- Routes de l'application Flask (restent les mêmes) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="Aucun fichier n'a été sélectionné.")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="Aucun fichier n'a été sélectionné.")

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        mask_image, error = get_segmentation_mask(filepath)

        if error:
            os.remove(filepath)
            return render_template('index.html', error=f"Erreur lors de la segmentation : {error}")

        mask_filename = "mask_" + filename
        mask_filepath = os.path.join(app.config['MASK_FOLDER'], mask_filename)
        if mask_image.mode != 'RGB':
            mask_image = mask_image.convert('RGB')
        mask_image.save(mask_filepath)

        uploaded_image_url = url_for('static', filename='images/uploads/' + filename)
        mask_image_url = url_for('static', filename='images/masks/' + mask_filename)

        return render_template('index.html',
                               uploaded_image=uploaded_image_url,
                               mask_image=mask_image_url,
                               message="Segmentation réussie !")
    return render_template('index.html', error="Erreur lors de l'upload.")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

# --- Démarrage de l'application Flask ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
