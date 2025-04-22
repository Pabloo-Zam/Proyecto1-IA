import numpy as np
from collections import defaultdict
import re

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)  # quitar URLs
    texto = re.sub(r"@\w+", "", texto)     # quitar menciones
    texto = re.sub(r"#\w+", "", texto)     # quitar hashtags
    texto = re.sub(r"[^\w\s]", "", texto)  # quitar puntuacion
    texto = re.sub(r"\d+", "", texto)      # quitar numeros
    texto = re.sub(r"[^a-zA-Zñáéíóúü\s]", "", texto)  # conserva letras y espacios
    texto = re.sub(r"\s+", " ", texto).strip()
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