# Application de Segmentation d'Image avec Flask et UNet-ResNet50

Cette application web permet de télécharger une image et d'obtenir son masque de segmentation sémantique en utilisant un modèle UNet avec un encodeur ResNet50 pré-entraîné sur le jeu de données Cityscapes (ou similaire, selon l'entraînement de votre modèle).

## Structure du Projet
image-segmentation-app/
├── app.py                  # Application Flask principale
├── model/                  # Contient le fichier de poids du modèle (.h5)
│   └── unet_resnet50_cityscapes.h5
├── static/                 # Fichiers CSS, JS, et dossiers pour les images uploadées/masques
│   ├── css/
│   ├── js/
│   └── images/
│       ├── uploads/
│       └── masks/
├── templates/              # Fichiers HTML des vues
│   └── index.html
├── requirements.txt        # Dépendances Python
├── Procfile                # Configuration pour le déploiement sur Render
├── .gitignore              # Fichiers à ignorer par Git
├── README.md               # Ce fichier
└── model_architecture.py   # Code pour reconstruire l'architecture du modèle UNet-ResNet50

## Configuration et Exécution Locale

### Prérequis

* Python 3.x
* pip

### Étapes

1.  **Clonez le dépôt Git** (ou créez la structure si vous n'avez pas de dépôt).
2.  **Placez votre modèle `.h5`** : Copiez le fichier `unet_resnet50_cityscapes.h5` (ou le nom réel de votre modèle) dans le dossier `model/`.
3.  **Assurez-vous que `model_architecture.py` contient la définition exacte de `create_unet_resnet50`** depuis votre notebook d'entraînement.
4.  **Créez un environnement virtuel (recommandé)** :
    ```bash
    python -m venv venv
    source venv/bin/activate # Sur Linux/macOS
    # venv\Scripts\activate # Sur Windows
    ```
5.  **Installez les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```
6.  **Exécutez l'application Flask** :
    ```bash
    python app.py
    ```
    L'application sera accessible à `http://127.0.0.1:5000/` (ou un autre port si spécifié).

## Déploiement sur Render

Cette application est configurée pour un déploiement facile sur Render.

1.  **Créez un compte Render** si vous n'en avez pas.
2.  **Liez votre dépôt GitHub** (où vous avez poussé ce code) à Render.
3.  **Créez un nouveau "Web Service"** sur Render.
4.  Configurez-le avec les paramètres suivants :
    * **Build Command**: `pip install -r requirements.txt`
    * **Start Command**: `gunicorn app:app`
    * Assurez-vous que le **Runtime** est `Python 3`.
    * Sélectionnez un plan (le plan gratuit peut être suffisant pour les tests, mais avec des limitations).
5.  Render construira et déploiera automatiquement votre application.

## Avertissement sur la Palette de Couleurs

La `CITYSCAPES_COLOR_PALETTE` définie dans `app.py` est une palette générique pour Cityscapes. Si votre modèle a été entraîné avec un mappage de classes ou une palette de couleurs différente, vous devrez ajuster `CITYSCAPES_COLOR_PALETTE` en conséquence pour une visualisation correcte.