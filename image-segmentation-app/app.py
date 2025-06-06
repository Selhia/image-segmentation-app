import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from skimage.color import label2rgb # Pour une meilleure visualisation des masques multi-classes

# Importer la fonction d'architecture du modèle depuis le nouveau fichier
from model_architecture import create_unet_resnet50

app = Flask(__name__)

# --- Configuration des dossiers ---
UPLOAD_FOLDER = 'static/images/uploads'
MASK_FOLDER = 'static/images/masks'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER

# Créer les dossiers s'ils n'existent pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# --- Paramètres du modèle (selon votre notebook) ---
MODEL_PATH = 'model/unet_resnet50_cityscapes.h5'
INPUT_SHAPE = (256, 512, 3) # Hauteur, Largeur, Canaux (confirmé par le notebook)
NUM_CLASSES = 8            # Nombre de classes (confirmé par le notebook)

# --- Palette de couleurs pour Cityscapes (exemple, à ajuster si votre mapping diffère) ---
# Ces couleurs sont les couleurs standard de Cityscapes, utiles pour la visualisation
# La documentation de Cityscapes fournit une liste exhaustive des classes et de leurs couleurs.
# Assurez-vous que l'ordre des couleurs correspond à l'ordre des classes de votre modèle.
CITYSCAPES_COLOR_PALETTE = [
    (0, 0, 0),        # 0: arrière-plan / void (ou non pertinent)
    (128, 64, 128),   # 1: route
    (244, 35, 232),   # 2: trottoir
    (70, 70, 70),     # 3: bâtiment
    (102, 102, 156),  # 4: mur
    (190, 153, 153),  # 5: clôture
    (153, 153, 153),  # 6: poteau
    (250, 170, 30),   # 7: panneau de signalisation
    # Ajoutez d'autres couleurs si vous avez plus de classes définies par votre modèle
    # Par exemple, pour les 19 classes courantes de Cityscapes, vous auriez besoin de plus:
    # (220, 220, 0),  # 8: lumière
    # (107, 142, 35), # 9: végétation
    # etc.
]

# --- Chargement du modèle ---
model = None
try:
    # Reconstruire l'architecture comme dans le notebook
    model = create_unet_resnet50(input_shape=INPUT_SHAPE, n_classes=NUM_CLASSES)
    # Charger les poids
    model.load_weights(MODEL_PATH)
    # Compiler le modèle (non strictement nécessaire pour la prédiction seule, mais bonne pratique)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Modèle '{MODEL_PATH}' chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    print("Vérifiez 'model_architecture.py' et la présence de 'unet_resnet50_cityscapes.h5'.")
    model = None # S'assurer que le modèle est None en cas d'échec

# --- Fonction de segmentation ---
def get_segmentation_mask(image_path, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1])):
    if model is None:
        return None, "Erreur: Le modèle n'a pas pu être chargé. Veuillez vérifier le serveur."

    try:
        # Charger et prétraiter l'image pour qu'elle corresponde à l'entrée du modèle
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Ajoute la dimension du batch
        img_array = img_array / 255.0 # Normalisation (si votre modèle a été entraîné avec cette normalisation)

        # Effectuer la prédiction
        prediction = model.predict(img_array)[0] # Récupère le masque prédit pour la première (et unique) image du batch

        # Post-traitement: Obtient l'indice de classe pour chaque pixel (np.argmax comme dans le notebook)
        segmentation_map = np.argmax(prediction, axis=-1)

        # Conversion du masque de segmentation en une image RGB colorée
        # Utilisez la palette définie pour Cityscapes.
        # `bg_label=0` indique que la classe 0 est l'arrière-plan et ne devrait pas être colorée par label2rgb
        # L'image originale `img_array[0]` peut être passée pour superposer le masque.
        # `colors=np.array(CITYSCAPES_COLOR_PALETTE)/255.0` normalise les couleurs pour skimage.
        colored_mask = label2rgb(
            segmentation_map,
            colors=np.array(CITYSCAPES_COLOR_PALETTE)/255.0,
            bg_label=0, # Si la classe 0 est l'arrière-plan/void
            image=img_array[0], # L'image originale pour la superposition
            alpha=0.5 # Opacité du masque superposé (ajuster selon préférence)
        )
        colored_mask = (colored_mask * 255).astype(np.uint8) # Convertir en format 0-255

        mask_image = Image.fromarray(colored_mask)

        return mask_image, None # Retourne l'image du masque PIL et aucune erreur

    except Exception as e:
        return None, f"Erreur lors de la prédiction ou du traitement du masque : {e}"

# --- Routes de l'application Flask ---
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
            os.remove(filepath) # Supprime l'image uploadée si la segmentation échoue
            return render_template('index.html', error=f"Erreur lors de la segmentation : {error}")

        # Sauvegarder le masque généré
        mask_filename = "mask_" + filename
        mask_filepath = os.path.join(app.config['MASK_FOLDER'], mask_filename)
        if mask_image.mode != 'RGB': # S'assurer que l'image est en RGB avant de sauvegarder
            mask_image = mask_image.convert('RGB')
        mask_image.save(mask_filepath)

        # URLs pour les images à afficher dans le template
        uploaded_image_url = url_for('static', filename='images/uploads/' + filename)
        mask_image_url = url_for('static', filename='images/masks/' + mask_filename)

        return render_template('index.html',
                               uploaded_image=uploaded_image_url,
                               mask_image=mask_image_url,
                               message="Segmentation réussie !")
    return render_template('index.html', error="Erreur lors de l'upload.")

# Route pour servir les fichiers statiques (images, CSS, JS)
@app.route('/static/<path:filename>')
def serve_static(filename):
    # Permet de servir des fichiers depuis le dossier 'static'
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)


# --- Démarrage de l'application Flask ---
if __name__ == '__main__':
    # Utiliser 0.0.0.0 pour rendre l'application accessible depuis l'extérieur du conteneur Docker/Render
    # Utiliser la variable d'environnement PORT fournie par Render, ou 5000 par défaut
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))