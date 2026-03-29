import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import splitfolders

# 1. Split Data (70% Train, 20% Val, 10% Test)
if not os.path.exists('data/processed'):
    splitfolders.ratio('data/raw/Training', output="data/processed", seed=1337, ratio=(.7, .2, .1))

# 2. Data Generators
datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory('data/processed/train', target_size=(224, 224), batch_size=32, class_mode='categorical')
val_gen = datagen.flow_from_directory('data/processed/val', target_size=(224, 224), batch_size=32, class_mode='categorical')
test_gen = datagen.flow_from_directory('data/processed/test', target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# 3. Simple CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# 5. Save Model and Metrics
if not os.path.exists('models'): os.makedirs('models')
if not os.path.exists('outputs'): os.makedirs('outputs')

model.save('models/brain_tumor_model.h5')

# Save Accuracy Plot
plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend(); plt.savefig('outputs/accuracy.png')

# Save Confusion Matrix
Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(test_gen.classes, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(test_gen.class_indices.keys()))
plt.savefig('outputs/confusion_matrix.png')

print("✅ Training done! Model and Plots saved.")
import pandas as pd # Add this at the top of train_model.py

# ... after your confusion matrix code ...

# Generate the report as a dictionary
report = classification_report(test_gen.classes, y_pred, 
                                target_names=list(test_gen.class_indices.keys()), 
                                output_dict=True)

# Convert to a DataFrame and save it
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('outputs/classification_report.csv')
print("✅ Classification Report saved!")
