import os
import pandas as pd
import numpy as np
import cloudpickle  # Cambiar de pickle a cloudpickle
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split, KFold  # Importar KFold
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from naive_bayes import NaiveBayes, limpiar_texto
from deep_translator import GoogleTranslator

def traducir_a_ingles(texto):
    return GoogleTranslator(source='auto', target='en').translate(texto)

def cargar_dataset_y_entrenar_kfold(k=5):
    # 1. Cargar el dataset completo
    df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
    df = df[[0, 5]]
    df.columns = ['sentimiento', 'texto']

    # 2. Preprocesamiento
    df['sentimiento'] = df['sentimiento'].replace({0: 'negativo', 2: 'neutro', 4: 'positivo'})
    df['texto'] = df['texto'].apply(limpiar_texto)

    # 3. Tomar el 80% 
    df = df.sample(frac=0.8, random_state=1)

    X = df['texto'].values
    y = df['sentimiento'].values

    # 4. Aplicar K-Fold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    precision_scores = []
    recall_scores = []
    f1_scores = []

    fold = 1
    for train_index, test_index in kf.split(X):
        print(f"\n🔁 Fold {fold}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        modelo = NaiveBayes()
        modelo.entrenar(X_train, y_train)

        y_pred = [modelo.predecir(tweet) for tweet in X_test]

        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"Precisión: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        fold += 1

    # 5. Promedio final de métricas
    print("\n📊 Promedio de métricas tras K-Fold:")
    print(f"Precisión promedio: {np.mean(precision_scores):.4f}")
    print(f"Recall promedio: {np.mean(recall_scores):.4f}")
    print(f"F1 Score promedio: {np.mean(f1_scores):.4f}")

    # 6. Entrenar el modelo final con todo el 80% y guardarlo
    modelo_final = NaiveBayes()
    modelo_final.entrenar(X, y)

    with open("modelo_nb.pkl", "wb") as f:
        cloudpickle.dump(modelo_final, f)  # Cambiar pickle.dump por cloudpickle.dump

    return modelo_final

# Flask app
app = Flask(__name__)

try:
    with open("modelo_nb.pkl", "rb") as f:
        modelo = cloudpickle.load(f)  # Cambiar pickle.load por cloudpickle.load
except FileNotFoundError:
    modelo = cargar_dataset_y_entrenar_kfold(k=5)  # Usa K-Fold si no hay modelo

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predecir", methods=["POST"])
def predecir():
    tweet = request.form.get("tweet", "")
    tweet_traducido = traducir_a_ingles(tweet)
    tweet_limpio = limpiar_texto(tweet_traducido)
    resultado = modelo.predecir(tweet_limpio)
    return render_template("index.html", tweet=tweet, prediccion=resultado)

if __name__ == '__main__':
    app.run(debug=True)
