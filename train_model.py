import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall, Precision
import matplotlib.pyplot as plt

def train():
    # 1. Load Dataset
    print("Loading dataset...")
    df = pd.read_csv('diabetes_dataset.csv')

    # 2. Refine Features
    # Drop race:* and location
    columns_to_drop = [col for col in df.columns if 'race:' in col] + ['location']
    df_refine = df.drop(columns=columns_to_drop)

    # 3. Preprocessing: Encoding
    # gender: Female (dropped), Male, Other
    # smoking_history: No Info (dropped), current, ever, former, never, not current
    df_encoded = pd.get_dummies(df_refine, columns=['gender', 'smoking_history'], drop_first=True)

    X = df_encoded.drop('diabetes', axis=1)
    y = df_encoded['diabetes']

    # Save column names to ensure consistency in web app
    column_names = X.columns.tolist()
    with open('columns.pkl', 'wb') as f:
        pickle.dump(column_names, f)
    print(f"Columns saved: {column_names}")

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved.")

    # 6. Class Weights
    weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(weights))

    # 7. Model Architecture (As per user's code)
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', Recall(name='recall'), Precision(name='precision')]
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 8. Training
    print("Starting training...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )

    # 9. Save Model and History Plot
    model.save('model.keras')
    print("Model saved to model.keras")

    # Generate Training History Plots
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Grafik Loss Pelatihan')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Grafik Akurasi Pelatihan')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=80)
    plt.close()
    print("Training history saved as training_history.png")

    # Final evaluation
    eval_result = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nFinal Evaluation Results:")
    print(f"Accuracy: {eval_result[1]:.4f}")
    print(f"Recall: {eval_result[2]:.4f}")
    print(f"Precision: {eval_result[3]:.4f}")

if __name__ == "__main__":
    train()
