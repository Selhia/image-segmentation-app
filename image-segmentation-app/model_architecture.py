# (b) Modèle U-Net avec backbone ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def create_unet_resnet50(input_shape=(256, 512, 3), n_classes=8):
    # --- Encoder : ResNet50 pré-entraîné ---
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Skip connections
    skip1 = base_model.get_layer('conv1_relu').output     # 128 x 256
    skip2 = base_model.get_layer('conv2_block3_out').output  # 64 x 128
    skip3 = base_model.get_layer('conv3_block4_out').output  # 32 x 64
    skip4 = base_model.get_layer('conv4_block6_out').output  # 16 x 32

    # Bottom
    x = base_model.get_layer('conv5_block3_out').output     # 8 x 16

    # --- Decoder : upsampling + concat with skip connections ---
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same')(x)   # 8x16 -> 16x32
    x = layers.Concatenate()([x, skip4])
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)   # 16x32 -> 32x64
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)   # 32x64 -> 64x128
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)    # 64x128 -> 128x256
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    # Dernier upsampling : 128x256 -> 256x512
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)

    # Output layer
    outputs = layers.Conv2D(n_classes, 1, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs, name='U-Net_ResNet50')
    return model

# (c) Modèle U-Net avec backbone VGG16
def create_unet_vgg16(input_shape=(256, 512, 3), n_classes=8):
    # Encoder pré-entraîné VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # Points de saut dans l'encodeur (on prend les sorties de chaque pool)
    skip1 = base_model.get_layer('block1_conv2').output  # après pool1: 128x256
    skip2 = base_model.get_layer('block2_conv2').output  # après pool2: 64x128
    skip3 = base_model.get_layer('block3_conv3').output  # après pool3: 32x64
    skip4 = base_model.get_layer('block4_conv3').output  # après pool4: 16x32
    # Sortie du dernier bloc de l'encodeur
    x = base_model.get_layer('block5_conv3').output        # après pool5: 8x16

    # Décodeur
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same')(x)  # 8x16 -> 16x32
    x = layers.Concatenate()([x, skip4])
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)  # 16x32 -> 32x64
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)  # 32x64 -> 64x128
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)   # 64x128 -> 128x256
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    # Pour remonter à 256x512 (on suppose input 256x512)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)   # 128x256 -> 256x512
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)

    # Couche de sortie
    output = layers.Conv2D(n_classes, 1, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=output, name='U-Net_VGG16')
    return model
