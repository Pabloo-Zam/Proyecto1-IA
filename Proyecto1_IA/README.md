Descripción del Proyecto:
Este proyecto implementa un clasificador de sentimientos (positivo/negativo/neutro) para tweets utilizando el algoritmo Naive Bayes. La aplicación web, desarrollada con Flask, permite:
Analizar tweets en español (traduciéndolos automáticamente al inglés para el análisis)
Mostrar métricas de rendimiento del modelo (Precisión, Recall, F1-Score)
Mantener un historial de los últimos análisis realizados
Visualizar resultados con una interfaz intuitiva

Tecnologías Utilizadas:
Python 3 (Lenguaje principal)
Flask (Framework web)
Scikit-learn (Implementación de K-Fold para validación)
Cloudpickle (Serialización del modelo)
Deep Translator (Traducción de textos)
Pandas/Numpy (Procesamiento de datos)
HTML5/CSS3 (Interfaz de usuario)

Instalación:

Clonar repositorio y entrar en la carpeta:

bash
git clone [URL] && cd Proyecto1_IA
Crear entorno virtual (opcional):

bash
python -m venv venv && source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
Instalar dependencias:

bash
pip install flask pandas scikit-learn deep-translator cloudpickle


Ejecución:

bash
python Proyecto1_IA.py
Abrir en navegador: http://localhost:5000

Datos:

Modelo pre-entrenado incluido (modelo_nb.pkl)
Dataset original: 1.6M tweets (se usa 80% para entrenamiento)
