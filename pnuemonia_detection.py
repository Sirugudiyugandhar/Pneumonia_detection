#we did it in google colab,so first mount the drive in colab
#STEP 1 — IMPORTs
import os, glob, random, cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#STEP 2 — DEFINE DATASET PATH
BASE_PATH = "/content/drive/MyDrive/chest_xray"

#STEP 3 — DATA GENERATORS(with augmentation for training)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_PATH, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_PATH, "val"),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_PATH, "test"),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

#STEP 4 — BUILD MODEL(InceptionV3 transfer learning)
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(224,224,3))

#Freeze pretrained layers
for layer in base_model.layers:
    layer.trainable = False

#Add your custom classification head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

#STEP 5 — COMPILE MODEL
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

#STEP 6 — CALLBACKS
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

#STEP 7 — HANDLE CLASS IMBALANCE AND TRAIN MODEL

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#Compute class weights from training data
labels = train_generator.classes          
classes = np.unique(labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)

#Convert to dictionary (Keras expects this format)
class_weight = {int(c): float(w) for c, w in zip(classes, class_weights)}
print(" Computed class weights:", class_weight)
# Example output → {0: 1.44, 1: 0.50}

#Train model with class weighting
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[es, lr_reduce],
    class_weight=class_weight 
)

#STEP 8 — EVALUATE MODEL
test_loss, test_acc = model.evaluate(test_generator)
print(f" Test accuracy: {test_acc:.4f}")

#STEP 9 — SAVE MODEL TO GOOGLE DRIVE
model.save('/content/drive/MyDrive/pneumonia_inceptionv3.h5')
print(" Model saved to Google Drive!")

#STEP 10 — PLOT & SAVE TRAINING & VALIDATION GRAPHS
# 1 Plot and Save Accuracy Graph
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Training vs Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('/content/drive/MyDrive/accuracy_curve.png')
plt.show()

#2 Plot and Save Loss Graph
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Training vs Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('/content/drive/MyDrive/loss_curve.png')
plt.show()

print("Accuracy and Loss graphs saved to Google Drive!")
