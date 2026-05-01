import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import tensorflow as tf

def generate_plots():
    # 1. Load data and model
    df = pd.read_csv('diabetes_dataset.csv')
    columns_to_drop = [col for col in df.columns if 'race:' in col] + ['location']
    df_refine = df.drop(columns=columns_to_drop)
    df_encoded = pd.get_dummies(df_refine, columns=['gender', 'smoking_history'], drop_first=True)
    
    X = df_encoded.drop('diabetes', axis=1)
    y = df_encoded['diabetes']
    
    # We need to replicate the split to get the same test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = tf.keras.models.load_model('model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
        
    X_test_scaled = scaler.transform(X_test[columns])
    
    # 2. Confusion Matrix
    y_probs = model.predict(X_test_scaled)
    y_pred = (y_probs > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title('Confusion Matrix - Prediksi Diabetes')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Confusion Matrix saved as confusion_matrix.png")
    
    # 3. Threshold Analysis
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    for t in thresholds:
        y_pred_t = (y_probs > t).astype(int)
        prec = precision_score(y_test, y_pred_t, zero_division=0)
        rec = recall_score(y_test, y_pred_t)
        acc = accuracy_score(y_test, y_pred_t)
        results.append({'Threshold': t, 'Precision': prec, 'Recall': rec, 'Accuracy': acc})
    
    df_threshold = pd.DataFrame(results)
    df_threshold.to_csv('threshold_analysis.csv', index=False)
    
    # 4. Threshold Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_threshold['Threshold'], df_threshold['Precision'], marker='o', label='Precision', color='blue')
    plt.plot(df_threshold['Threshold'], df_threshold['Recall'], marker='s', label='Recall', color='red')
    plt.plot(df_threshold['Threshold'], df_threshold['Accuracy'], marker='^', label='Accuracy', color='green')
    plt.axvline(x=0.5, linestyle='--', color='gray', label='Default Threshold (0.5)')
    plt.title('Trade-off Metrik Berdasarkan Perubahan Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Skor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('threshold_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("Threshold analysis and plot saved.")
    print(df_threshold)

if __name__ == "__main__":
    generate_plots()
