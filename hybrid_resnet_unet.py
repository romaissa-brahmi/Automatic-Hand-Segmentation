# --------------------
# Import the libraries
# --------------------
print("⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡")
print("Importing the libraries...")

import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

from src.gpu_config import GPUManager
gpu_manager = GPUManager(gpu_ids="1")
strategy = gpu_manager.get_strategy()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from datetime import datetime

import tensorflow as tf


keras = tf.keras
layers = tf.keras.layers
Model = tf.keras.Model
metrics = tf.keras.metrics
utils = tf.keras.utils

import segmentation_models as sm

from sklearn.model_selection import train_test_split



# --------------------
# Create the variables
# --------------------
print("Creating the variables...")

EPOCHS_RESNET = 10
EPOCHS_UNET = 30
BATCH_SIZE = 32
learning_rate_resnet = 1e-3
learning_rate_unet = 1e-4

now = datetime.now()
date_now = now.strftime("%d/%m/%Y %H:%M:%S")
metrics_file_name = now.strftime("%d%m_%H%M")


# -------------
# Load the data
# -------------
def get_center(lm0, lm5, lm17):

    x = np.mean([lm0[0], lm5[0], lm17[0]])
    y = np.mean([lm0[1], lm5[1], lm17[1]])

    return float(x), float(y)


def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5):
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = (mask > 0).astype(np.uint8)
    color_image = np.full_like(image, color, dtype=np.uint8)
    blended = cv2.addWeighted(image, 1 - alpha, color_image, alpha, 0)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    result = np.where(mask_3d == 1, blended, image)

    return result.astype(np.uint8)



print("Loading the data...")
print("⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡ ⋆˚꩜｡")

input_shape = (128, 128)

X_directory = "data/images"
y_directory = "data/masks"
df = pd.read_csv("data/hand_data.csv")
df.columns = df.columns.str.strip() # y'a des espaces au début du nom de certaines colonnes


X, y_mask, y_landmarks, image_paths = [], [], [], []

for _, row in df.iterrows():

    id_image = int(row["image_index"])

    image_path = os.path.join(X_directory, f"{id_image}.png")
    mask_path = os.path.join(y_directory, f"{id_image}.png")

    image = cv2.imread(image_path)
    label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue
    if label is None:
        continue

    image_resized = cv2.resize(image, input_shape)
    label_resized = cv2.resize(label, input_shape, interpolation=cv2.INTER_NEAREST)

    label_resized = label_resized / 255.0
    label_resized = np.expand_dims(label_resized, axis=-1)

    h_original, w_original = image.shape[:2]
    lm0 = [float(row["pt0_x"]), float(row["pt0_y"])]
    lm5 = [float(row["pt5_x"]), float(row["pt5_y"])]
    lm17 = [float(row["pt17_x"]), float(row["pt17_y"])]
    center = get_center(lm0, lm5, lm17)

    lm_vector = np.array([
        lm0[0], lm0[1],
        lm5[0], lm5[1],
        lm17[0], lm17[1],
        center[0], center[1]
    ], dtype=np.float32)

    X.append(image_resized)
    y_mask.append(label_resized)
    y_landmarks.append(lm_vector)
    image_paths.append(id_image)

print(f"Chargement terminé : {len(X)} images chargées.")

X = np.array(X)
y_mask = np.array(y_mask)
y_landmarks = np.array(y_landmarks, dtype=np.float32)

print("X shape =", X.shape, "y_mask shape =", y_mask.shape, "y_landmarks shape =", y_landmarks.shape)


if np.min(X)>0 or np.max(X)>1:
    print(np.min(X))
    print(np.max(X))
    print("X n'est pas normalisé!")
elif len(np.unique(y_mask)>2):
    print(np.unique(y_mask))
    print("les masques ne sont pas binarisés!")


# -------------
# Preprocessing
# -------------
print("✩₊˚⊹.⋆☾⋆⁺₊✧ ✩₊˚⊹.⋆☾⋆⁺₊✧ ✩₊˚⊹.⋆☾⋆⁺₊✧ ✩₊˚⊹.⋆☾⋆⁺₊✧")
print("✩₊˚⊹.⋆☾⋆⁺₊✧      Preprocessing      ✩₊˚⊹.⋆☾⋆⁺₊✧")
print("✩₊˚⊹.⋆☾⋆⁺₊✧ ✩₊˚⊹.⋆☾⋆⁺₊✧ ✩₊˚⊹.⋆☾⋆⁺₊✧ ✩₊˚⊹.⋆☾⋆⁺₊✧")

preprocess_input = sm.get_preprocessing('resnet50')
X = preprocess_input(X)

X_train, X_test, y_mask_train, y_mask_test, y_lm_train, y_lm_test, paths_train, paths_test = train_test_split(X, y_mask, y_landmarks, image_paths, test_size=0.2, random_state=12)
print(f"Images ==> Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Masks  ==> Train samples: {y_mask_train.shape[0]}, Test samples: {y_mask_test.shape[0]}")
print(f"Landmarks  ==> Train samples: {y_lm_train.shape[0]}, Test samples: {y_lm_test.shape[0]}")



dataset = tf.data.Dataset.from_tensor_slices((X_train, {"seg": y_mask_train, "landmarks": y_lm_train}))
dataset = dataset.shuffle(buffer_size=1000)

train_size = int(0.8 * len(X_train))
training_dataset = dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

validation_dataset = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

testing_dataset = tf.data.Dataset.from_tensor_slices((X_test, {"seg": y_mask_test, "landmarks": y_lm_test}))
testing_dataset = testing_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



# ------------------------
# Build the Neural Network
# ------------------------
print("✮ ⋆ ˚｡𖦹 ⋆｡°✩ ✮ ⋆ ˚｡𖦹 ⋆｡°✩ ✮ ⋆ ˚｡𖦹 ⋆｡°✩ ✮ ⋆ ˚｡𖦹 ⋆｡°✩ ✮ ⋆ ˚｡𖦹 ⋆｡°✩")
print("✮ ⋆ ˚｡𖦹 ⋆｡°✩     Building the Neural Network...    ✮ ⋆ ˚｡𖦹 ⋆｡°✩")
print("✮ ⋆ ˚｡𖦹 ⋆｡°✩ ✮ ⋆ ˚｡𖦹 ⋆｡°✩ ✮ ⋆ ˚｡𖦹 ⋆｡°✩ ✮ ⋆ ˚｡𖦹 ⋆｡°✩ ✮ ⋆ ˚｡𖦹 ⋆｡°✩")

def unet_2d_multi_pretrained(input_shape=(128, 128, 3)):
    # 1) Load pre-trained unet
    base_model = sm.Unet(
        backbone_name='resnet50',
        encoder_weights='imagenet',
        encoder_freeze=True,
        input_shape=input_shape,
        classes=1,
        activation='sigmoid'
    )

    # 2) get unet bottleneck
    encoder_output = base_model.get_layer('relu1').output

    # 3) landmarks output
    x = layers.GlobalAveragePooling2D()(encoder_output)
    x = layers.Dense(512, activation='relu')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    lm_output = layers.Dense(8, activation='linear', name="landmarks")(x)

    # 4) segmentation output
    seg_output = layers.Layer(name="seg")(base_model.output)

    return Model(inputs=base_model.input, outputs=[seg_output, lm_output])


# --------
# Training
# --------
print("°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*")
print("°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*         Training the Neural Network...        °❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*")
print("°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*")


print("°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*         ResNet part...        °❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*")

with strategy.scope():
    model = unet_2d_multi_pretrained(input_shape=(128, 128, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_resnet),
        loss={
            "seg": sm.losses.bce_dice_loss,
            "landmarks": "mse"
        },

        loss_weights={
            "seg": 1.0,
            "landmarks": 10.0
        },

        metrics={
            "seg": [metrics.BinaryIoU(name="iou")],
            "landmarks": ["mae"]
        }
    )
    model.summary()

model.fit(training_dataset, validation_data=validation_dataset, epochs=EPOCHS_RESNET)


print("°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*         U-Net part...        °❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*")

model.trainable = True
with strategy.scope():

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_unet),
        loss={
            "seg": sm.losses.bce_dice_loss,
            "landmarks": "mse"
        },

        loss_weights={
            "seg": 1.0,
            "landmarks": 10.0
        },

        metrics={
            "seg": [metrics.BinaryIoU(name="iou")],
            "landmarks": ["mae"]
        }
    )

    model.summary()



history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_UNET
)


fig, axs = plt.subplots(1, 2, figsize=(15, 5))

fig.suptitle(f"Metrics - {date_now}", fontsize=16)

# --- Segmentation (IoU)
axs[0].plot(history.history['seg_iou'], label='Train IoU')
axs[0].plot(history.history['val_seg_iou'], label='Val IoU', linestyle='--')
axs[0].set_title('Segmentation IoU')
axs[0].legend()
axs[0].grid(True)

# --- Landmarks (MAE)
axs[1].plot(history.history['landmarks_mae'], label='Train MAE')
axs[1].plot(history.history['val_landmarks_mae'], label='Val MAE', linestyle='--')
axs[1].set_title('Landmarks Error')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()

# --- Sauvegarde du graphique ---
os.makedirs("metric_results", exist_ok=True)
plt.savefig(f"metric_results/metrics_hybrid_resnet_{metrics_file_name}.png")



# ----------
# Evaluating
# ----------
print("✩₊˚.⋆☾⋆⁺₊✧ ✩₊˚.⋆☾⋆⁺₊✧ ✩₊˚.⋆☾⋆⁺₊✧ ✩₊˚.⋆☾⋆⁺₊✧ ✩₊˚.⋆☾⋆⁺₊✧ ✩₊˚.⋆☾⋆⁺₊✧")
print("Evaluating the Neural Network...")
results = model.evaluate(testing_dataset)
results_dict = dict(zip(model.metrics_names, results))

# -------
# Predict
# -------
print("Predicting...")
print("✩₊˚.⋆☾⋆⁺₊✧ ✩₊˚.⋆☾⋆⁺₊✧ ✩₊˚.⋆☾⋆⁺₊✧ ✩₊˚.⋆☾⋆⁺₊✧ ✩₊˚.⋆☾⋆⁺₊✧ ✩₊˚.⋆☾⋆⁺₊✧")

os.makedirs("output_hybrid_resnet", exist_ok=True)

pred_seg, pred_lm = model.predict(X_test, batch_size=BATCH_SIZE)

for i, (image_BGR, predicted_mask, true_mask, pred_points, true_points) in enumerate(
        zip(X_test, pred_seg, y_mask_test, pred_lm, y_lm_test)):

    id_image = paths_test[i]

    image_path = os.path.join(X_directory, f"{id_image}.png")
    mask_path = os.path.join(y_directory, f"{id_image}.png")

    image_original = cv2.imread(image_path)
    mask_original = cv2.imread(mask_path)
    image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    h, w = image_original.shape[:2]

    mask_resized = cv2.resize(predicted_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_smoothed = cv2.GaussianBlur(mask_resized, (15, 15), 0)
    binary_mask = (mask_smoothed > 0.5).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

    p_points = pred_points.reshape(4, 2)
    p_landmarks = p_points[:3]
    p_center = p_points[3]

    t_points = true_points.reshape(4, 2)
    t_landmarks = t_points[:3]
    t_center = t_points[3]

    # --- Affichage Image + Landmarks ---
    axs[0].imshow(image_original)

    for j, (x, y) in enumerate(t_landmarks):
        axs[0].scatter(x * w, y * h, c='salmon', s=80, edgecolors='white', marker='o',
                       label='True Landmarks' if j == 0 else "")

    axs[0].scatter(t_center[0] * w, t_center[1] * h, c='lightgreen', s=100, edgecolors='white', marker='X',
                   label='True Center')

    for j, (x, y) in enumerate(p_landmarks):
        axs[0].scatter(x * w, y * h, c='red', s=80, edgecolors='white', marker='o',
                       label='Pred Landmarks' if j == 0 else "")

    axs[0].scatter(p_center[0] * w, p_center[1] * h, c='lime', s=100, edgecolors='white', marker='X',
                   label='Pred Center')

    axs[0].legend(loc='upper right', fontsize='12')
    axs[0].set_xlim(0, w)
    axs[0].set_ylim(h, 0)
    axs[0].set_title("Landmarks: Pred vs True", fontsize='17')
    axs[0].axis('off')

    # --- Carte de segmentation prédite ---
    predicted_mask_image = overlay_mask_on_image(image_original, binary_mask, (200,0,0))
    axs[1].imshow(predicted_mask_image)
    axs[1].set_title("Predicted mask", fontsize='17')
    axs[1].axis('off')

    # --- Vraie carte de segmentation ---
    true_mask_image = overlay_mask_on_image(image_original, mask_original, (0,200,0))
    axs[2].imshow(true_mask_image)
    axs[2].set_title("True mask", fontsize='17')
    axs[2].axis('off')

    plt.tight_layout()
    fig.savefig(f'output_hybrid_resnet/{i}.png', bbox_inches='tight')
    plt.close(fig)


# ----------------
# Save the results
# ----------------
new_data = {
    "Date_Execution": date_now,
    "Input shape": input_shape,
    "NB_IMAGES": X.shape[0],
    "Learning Rate Resnet": learning_rate_resnet,
    "Learning Rate Unet": learning_rate_unet,
    "Batch Size": BATCH_SIZE,
    "Epochs Resnet": EPOCHS_RESNET,
    "Epochs Unet": EPOCHS_UNET,

    "Training IoU": max(history.history['seg_iou']) * 100,
    "Validation IoU": max(history.history['val_seg_iou']) * 100,
    "Test IoU": results_dict['seg_iou'] * 100,

    "Training MAE": min(history.history['landmarks_mae']),
    "Validation MAE": min(history.history['val_landmarks_mae']),
    "Test MAE": results_dict['landmarks_mae'],

    "Training Loss": min(history.history['loss']),
    "Validation Loss": min(history.history['val_loss']),
    "Test Loss": results_dict['loss'],

    "Test Seg Loss": results_dict['seg_loss'],
    "Test Landmarks Loss": results_dict['landmarks_loss'],
}

df_new = pd.DataFrame([new_data])

df_new = df_new.round({
    "Training IoU": 2,
    "Validation IoU": 2,
    "Test IoU": 2,

    "Training MAE": 4,
    "Validation MAE": 4,
    "Test MAE": 4,

    "Training Loss": 4,
    "Validation Loss": 4,
    "Test Loss": 4,

    "Test Seg Loss": 4,
    "Test Landmarks Loss": 4,
})


CSV_FILE = "metric_results/results_u_net_hybrid_resnet.csv"

if os.path.exists(CSV_FILE):
    df_existing = pd.read_csv(CSV_FILE)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(CSV_FILE, index=False)
else:
    df_new.to_csv(CSV_FILE, index=False)

df = pd.read_csv(CSV_FILE)

print(". ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.")
print(". ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.      Program done running !        . ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.")
print(". ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.")
