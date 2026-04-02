# --------------------
# Import the libraries
# --------------------
print("Importing the libraries...")

import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from datetime import datetime

import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

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
NB_IMAGES = 11
EPOCHS = 20
BATCH_SIZE = 32
learning_rate = 0.001

now = datetime.now()
date_now = now.strftime("%d/%m/%Y %H:%M:%S")
metrics_file_name = now.strftime("%Y%m%d_%H%M%S")


# -------------
# Load the data
# -------------
def parse_coord(coord_str):
    x, y = coord_str.split("x")
    return float(x), float(y)


print("Loading the data...")
input_shape = (96, 96)

X_directory = "data/images"
y_directory = "data/masks"

df = pd.read_csv("landmarks_data.csv")
df.columns = df.columns.str.strip() # y'a des espaces au début du nom de certaines colonnes
#df = df.head(NB_IMAGES)

X, y_mask, y_landmarks = [], [], []

for _, row in df.iterrows():
    id_str = str(row["id"]).zfill(6)

    image_path = os.path.join(X_directory, f"{id_str}.png")
    mask_path = os.path.join(y_directory, f"{id_str}.png")

    image = cv2.imread(image_path)
    label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Image introuvable ou illisible : {image_path}")
        continue
    if label is None:
        print(f"Masque introuvable ou illisible : {mask_path}")
        continue

    image_resized = cv2.resize(image, input_shape)
    label_resized = cv2.resize(label, input_shape, interpolation=cv2.INTER_NEAREST)

    image_resized = image_resized / 255.0
    label_resized = label_resized / 255.0
    label_resized = np.expand_dims(label_resized, axis=-1)


    lm0 = parse_coord(row["0"])
    lm5 = parse_coord(row["5"])
    lm17 = parse_coord(row["17"])
    center = parse_coord(row["0-5-17 center"])

    lm_vector = np.array([*lm0, *lm5, *lm17, *center], dtype=np.float32)

    X.append(image_resized)
    y_mask.append(label_resized)
    y_landmarks.append(lm_vector)

print(f"Chargement terminé : {len(X)} images chargées.")

X = np.array(X)
y_mask = np.array(y_mask)
y_landmarks = np.array(y_landmarks, dtype=np.float32)

if len(y_landmarks.shape) == 1:
    y_landmarks = np.stack(y_landmarks)

h, w = input_shape

y_landmarks[:, 0::2] = y_landmarks[:, 0::2] / w
y_landmarks[:, 1::2] = y_landmarks[:, 1::2] / h

print("X shape =", X.shape, "y_mask shape =", y_mask.shape, "y_landmarks shape =", y_landmarks.shape)


# -------------
# Preprocessing
# -------------
print("Preprocessing...")

X_train, X_test, y_mask_train, y_mask_test, y_lm_train, y_lm_test = train_test_split(X, y_mask, y_landmarks, test_size=0.2, random_state=12)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Train samples: {X_test.shape[0]}, Test samples: {y_mask_test.shape[0]}")

dataset = tf.data.Dataset.from_tensor_slices((X_train, {"seg": y_mask_train,
                                                        "landmarks": y_lm_train
                                                       }))

dataset = dataset.shuffle(buffer_size=1000)

train_size = int(0.8 * len(X_train))

training_dataset = dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

testing_dataset = tf.data.Dataset.from_tensor_slices((X_test,{"seg": y_mask_test,
                                                              "landmarks": y_lm_test
                                                             })).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ------------------------
# Build the Neural Network
# ------------------------
print("Building the Neural Network...")
def unet_2d_multi(input_shape=(96, 96, 3)):
    inputs = layers.Input(shape=input_shape)

    # ------ Encoder ------
    c1 = layers.Conv2D(8, 3, activation='relu', padding='same')(inputs)
    c2 = layers.Conv2D(8, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c2)
    

    c3 = layers.Conv2D(16, 3, activation='relu', padding='same')(p1)
    c4 = layers.Conv2D(16, 3, activation='relu', padding='same')(c3)
    p2 = layers.MaxPooling2D(2)(c4)

    # ------ Bottleneck ------
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(p2)

    # ------ Decoder ------

    u1 = layers.UpSampling2D(2)(c5)
    u1 = layers.Concatenate()([u1, c4])
    u2 = layers.UpSampling2D(2)(u1)
    u2 = layers.Concatenate()([u2, c2])

    # ------ Output layers ------
    seg_output = layers.Conv2D(1, 1, activation='sigmoid', name="seg")(u2)
    flat = layers.GlobalAveragePooling2D()(c5)
    lm_output = layers.Dense(8, activation='sigmoid', name="landmarks")(flat)

    return Model(inputs, [seg_output, lm_output])


model = unet_2d_multi(input_shape=(96, 96, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss={
                  "seg": sm.losses.DiceLoss(),
                  "landmarks": "mse"
              },
              metrics={
                        "seg": [metrics.BinaryIoU(name="iou")],
                        "landmarks": ["mae"]
              }
             )
model.summary()


# --------
# Training
# --------
print("Training the Neural Network...")
history = model.fit(X_train,{"seg": y_mask_train,
                             "landmarks": y_lm_train
                            },
          validation_data=(X_test,{"seg": y_mask_test,
                                   "landmarks": y_lm_test }),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE
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
plt.savefig(f"metric_results/metrics_{metrics_file_name}.png")

plt.show()


# ----------
# Evaluating
# ----------
print("Evaluating the Neural Network...")
results = model.evaluate(X_test,{"seg": y_mask_test,
                                 "landmarks": y_lm_test
                                })

results_dict = dict(zip(model.metrics_names, results))


# -------
# Predict
# -------
print("Predicting...")
os.makedirs("output", exist_ok=True)

pred_seg, pred_lm = model.predict(X_test, batch_size=BATCH_SIZE)

for i, (image_BGR, predicted_mask, true_mask, pred_points) in enumerate(zip(X_test, pred_seg, y_mask_test, pred_lm)):

    binary_mask = (predicted_mask > 0.5).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    image_BGR = (image_BGR * 255).astype('uint8')
    image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

    # --- Images et landmarks ---
    axs[0].imshow(image_RGB)

    h, w = image_RGB.shape[:2]

    points = pred_points.reshape(4, 2)

    landmarks = points[:3]
    center_pt = points[3]

    for (x, y) in landmarks:
        px = x * w
        py = y * h
        axs[0].scatter(px, py, c='red', s=40, edgecolors='white', marker='o', label='Landmarks' if (x == landmarks[0][0]) else "")

    cx = center_pt[0] * w
    cy = center_pt[1] * h
    axs[0].scatter(cx, cy, c='lime', s=60, edgecolors='white', marker='X', label='Center')

    axs[0].legend(loc='upper right', fontsize='xx-small')
    axs[0].set_xlim(0, w)
    axs[0].set_ylim(h, 0)
    axs[0].set_title("Image + Pred landmarks")
    axs[0].axis('off')

    # --- Carte de segmentation prédite ---
    axs[1].imshow(binary_mask, cmap='gray')
    axs[1].set_title("Predicted mask")
    axs[1].axis('off')

    # --- Vraie carte de segmentation ---
    axs[2].imshow(true_mask, cmap='gray')
    axs[2].set_title("True mask")
    axs[2].axis('off')

    plt.tight_layout()
    fig.savefig(f'output/{i}.png')
    plt.close(fig)


# ----------------
# Save the results
# ----------------
new_data = {
    "Date_Execution": date_now,
    "Input shape": input_shape,
    "NB_IMAGES": NB_IMAGES,
    "Learning Rate": learning_rate,
    "Batch Size": BATCH_SIZE,
    "Epochs": EPOCHS,

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

if os.path.exists("results_u_net.csv"):
    df_existing = pd.read_csv("results_u_net.csv")
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv("results_u_net.csv", index=False)
else:
    df_new.to_csv("results_u_net.csv", index=False)

df = pd.read_csv("results_u_net.csv")

print("Program done running!")