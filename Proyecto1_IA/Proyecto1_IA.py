# Proyecto1_IA.py

import os
import pandas as pd
import numpy as np
import re
import string
import pickle
from collections import defaultdict
from flask import Flask, request, render_template

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)  # quitar URLs
    texto = re.sub(r"@\w+", "", texto)     # quitar menciones
    texto = re.sub(r"#\w+", "", texto)     # quitar hashtags
    texto = re.sub(r"[^\w\s]", "", texto)  # quitar puntuacion
    texto = re.sub(r"\d+", "", texto)      # quitar numeros
    texto = texto.strip()
    return texto

class NaiveBayes:
    def __init__(self):
        self.clases = None
        self.frecuencias_clase = {}
        self.frecuencias_palabra = {}
        self.vocabulario = set()
        self.total_palabras_clase = {}

    def entrenar(self, X, y):
        self.clases = np.unique(y)
        self.frecuencias_clase = defaultdict(int)
        self.frecuencias_palabra = {c: defaultdict(int) for c in self.clases}
        self.total_palabras_clase = {c: 0 for c in self.clases}

        for tweet, clase in zip(X, y):
            palabras = tweet.split()
            self.frecuencias_clase[clase] += 1
            for palabra in palabras:
                self.frecuencias_palabra[clase][palabra] += 1
                self.vocabulario.add(palabra)
                self.total_palabras_clase[clase] += 1


    def predecir(self, texto):
        palabras = texto.split()
        log_probs = {}
        total_docs = sum(self.frecuencias_clase.values())

        for clase in self.clases:
            log_probs[clase] = np.log(self.frecuencias_clase[clase] / total_docs)
            for palabra in palabras:
                conteo = self.frecuencias_palabra[clase].get(palabra, 0) + 1
                total = self.total_palabras_clase[clase] + len(self.vocabulario)
                log_probs[clase] += np.log(conteo / total)

        return max(log_probs, key=log_probs.get)


def cargar_dataset_y_entrenar():
    
    df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
    df = df[[0, 5]]
    df.columns = ['sentimiento', 'texto']

    
    df = df.sample(n=10000, random_state=1)

    
    df['sentimiento'] = df['sentimiento'].replace({0: 'negativo', 2: 'neutro', 4: 'positivo'})
    df['texto'] = df['texto'].apply(limpiar_texto)

    X = df['texto'].values
    y = df['sentimiento'].values

    # 🔀 Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    modelo = NaiveBayes()
    modelo.entrenar(X_train, y_train)

    y_pred = [modelo.predecir(tweet) for tweet in X_test]

    
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    
    with open("modelo_nb.pkl", "wb") as f:
        pickle.dump(modelo, f)

    
    return modelo




app = Flask(__name__)

try:
    with open("modelo_nb.pkl", "rb") as f:
        modelo = pickle.load(f)
except FileNotFoundError:
   # modelo = cargar_dataset_y_entrenar()
       raise RuntimeError("El modelo no se encontró. Debes entrenarlo localmente y subir modelo_nb.pkl.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    tweet = request.form['tweet']
    tweet_limpio = limpiar_texto(tweet)
    resultado = modelo.predecir(tweet_limpio)
    return render_template('index.html', prediccion=resultado, tweet=tweet)

if __name__ == '__main__':
    app.run(debug=True)

   


