import tensorflow as tf
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# from google.colab import drive, files
# drive.mount('/content/drive')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_colwidth', None)

def cargar_imagen_redimensionada(ruta_imagen, ancho=224, alto=224):
    datos_imagen = tf.io.read_file(ruta_imagen)
    imagen = tf.io.decode_jpeg(datos_imagen)
    imagen = tf.image.resize(imagen, [alto, ancho])
    imagen = tf.cast(imagen, tf.float32)
    return imagen

def graficar_imagen(ruta_imagen, titulo=None, guardar_en=None, figsize=(4, 7)):
    try:
        imagen = cargar_imagen_redimensionada(ruta_imagen)
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(imagen.numpy().astype(np.uint8))
    if titulo is None:
        titulo = f"Imagen: {os.path.basename(ruta_imagen)}"
    ax.set_title(titulo)
    ax.axis('off')
    if guardar_en:
        plt.savefig(guardar_en, dpi=300, bbox_inches='tight')
        print(f"Imagen guardada en: {guardar_en}")

    plt.tight_layout()
    plt.show()

def especie_batch(data):
    species = list(set(data["label"]))
    num_species = len(species)
    for i, specie in enumerate(tqdm(species, desc=f"Especies procesadas")):
        print(f"\nEspecie {i+1} de {(num_species)}, ({specie})")
        yield specie

def specie_ds(data, gen, preprocessor):
    try:
        specie = next(gen)
        df = data[data["label"] == specie]
        if df.empty:
            print(f"No more species")
            return None, None
    except StopIteration:
        return None, None
    return specie, preprocessor.create_validation_dataset(df)

def calcular_incertidumbre(dataset_espectrogramas, modelo):
    predicciones_mc = []
    confianzas_mc = []
    etiqueta_real_lista =[]
    etiqueta_inferencia_lista = []
    probabilidades_todas = []  # Lista para almacenar todas las probabilidades

    if dataset_espectrogramas:
        for imagenes_batch, etiquetas_batch in dataset_espectrogramas.take(1):

            imagen = imagenes_batch[0]
            imagen_batch = tf.expand_dims(imagen, axis=0)
            etiqueta_real = tf.argmax(etiquetas_batch[0])
            etiqueta_real_lista.append(etiqueta_real.numpy())
            # Predicción convencional (sin dropout activo)
            logits_inferencia = modelo(imagen_batch, training=False)
            #logits_inferencia = modelo.predict(imagen_batch)
            probabilidades_inferencia = tf.nn.softmax(logits_inferencia, axis=-1).numpy()
            clase_predicha_inferencia = np.argmax(probabilidades_inferencia, axis=-1)[0]
            etiqueta_inferencia_lista.append(clase_predicha_inferencia)

            # MONTE CARLO DROPOUT: Múltiples pasadas con training=True
            for _ in range(2000):
                logits = modelo(imagen_batch, training=True)
                probabilidades = tf.nn.softmax(logits, axis=-1).numpy()
                probabilidades_todas.append(probabilidades[0])  # Guardamos todas las probabilidades
                clase_predicha = np.argmax(probabilidades, axis=-1)[0]
                confianza = probabilidades[0, clase_predicha]

                predicciones_mc.append(clase_predicha)
                confianzas_mc.append(confianza)

    # Calcular la entropía si hay datos disponibles
    entropia = 0
    if probabilidades_todas:
        # Calculamos la entropía promedio
        probabilidades_array = np.array(probabilidades_todas)
        prob_media = np.mean(probabilidades_array, axis=0)
        entropia = -np.sum(prob_media * np.log(prob_media + 1e-9))

    return predicciones_mc, confianzas_mc, etiqueta_inferencia_lista, etiqueta_real_lista, entropia

def crear_dataframe_incertidumbre(data, muestra, modelo, preprocessor):
    # Generar dataset de especies
    dataset_generado = especie_batch(muestra)

    # Recolectar resultados de cada especie
    resultados = {}
    for i in range(data["label"].nunique()):
        especie, dataset_especie = specie_ds(data, dataset_generado, preprocessor)
        if especie is None:
            break

        predicciones_mc, confianzas_mc, clase_predicha_inferencia, etiqueta_real, entropia = calcular_incertidumbre(dataset_especie, modelo)
        resultados[especie] = {
            "clase_real": etiqueta_real,
            "clase_predicha_inferencia": clase_predicha_inferencia,
            "predicciones_mc": predicciones_mc,
            "confianzas_mc": confianzas_mc,
            "entropia": entropia
        }

    # Procesar resultados para crear el dataframe final
    filas = []
    for especie, datos in resultados.items():
        clase_real = datos["clase_real"]  # Lista de etiquetas reales
        clase_predicha = datos["clase_predicha_inferencia"]  # Lista de predicciones normales
        predicciones = datos["predicciones_mc"]  # Lista de predicciones Monte Carlo
        confianzas = datos["confianzas_mc"]  # Lista de confianzas
        entropia = datos["entropia"]

        # Para cada predicción Monte Carlo
        for i in range(len(predicciones)):
            # Etiqueta real debe ser consistente para todas las filas de la misma imagen
            etiqueta_real = clase_real[0] if len(clase_real) > 0 else None
            # Clase predicha normal (solo una por imagen)
            etiqueta_pred = clase_predicha[0] if len(clase_predicha) > 0 else None

            filas.append({
                "especie": especie,
                "clase_real": etiqueta_real,
                "clase_predicha": etiqueta_pred,
                "prediccion_mc": predicciones[i],
                "confianza_mc": confianzas[i] if i < len(confianzas) else None,
                # Comparación correcta de valores individuales, no listas
                'true_positive': 1 if etiqueta_real == etiqueta_pred else 0,
                'true_positivo_mc': 1 if etiqueta_real == predicciones[i] else 0,
                "entropia": entropia
            })

    return pd.DataFrame(filas)

def plot_incertidumbres(especie_df, label, title):
    plt.figure(figsize=(7, 4))
    sns.set_theme(style="white")

    # Crear un array con los colores - rojo para todas las barras
    colores = ['#FF4136'] * len(especie_df)  # Color rojo para todas las barras

    especie_df = especie_df.sort_values(by="prediccion_mc", ascending=True).reset_index(drop=True)
    indice_label = especie_df[especie_df['prediccion_mc'] == label].index

    for idx in indice_label:
        colores[idx] = '#2ECC40'  # Color verde para las barras con label 300

    # Crear el gráfico con los colores personalizados
    ax = sns.barplot(
        x=especie_df['prediccion_mc'],
        y=especie_df["proportion"],
        palette=colores,  # Usar la lista de colores personalizada
        edgecolor="black",
        linewidth=0.8
    )

    # Añadir título y etiquetas
    plt.title(f'Distribución de predicciones para {title}',
            fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Clase predicha', fontsize=12)
    plt.ylabel('Proporción', fontsize=12)

    # Mejorar la cuadrícula
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Añadir valores sobre las barras principales
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        height_100 = height * 100
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                    f'{height_100:.1f}%', ha="center", fontsize=10)

    plt.ylim(0,1)

    # Eliminar líneas del marco excepto abajo y a la izquierda
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_combinado(ruta_imagen, especie_df, label, titulo_especie, figsize=(12, 5)):
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Primer subplot: Mostrar imagen
    try:
        imagen = cargar_imagen_redimensionada(ruta_imagen)
        ax1.imshow(imagen.numpy().astype(np.uint8))
        ax1.set_title(f"Spectogram: {titulo_especie}")
        ax1.axis('off')
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        ax1.text(0.5, 0.5, "Error al cargar la imagen",
                 ha='center', va='center')
        ax1.axis('off')

    # Segundo subplot: Gráfico de incertidumbres
    plt.sca(ax2)  # Establecer ax2 como el eje activo

    # Ordenar datos
    especie_df = especie_df.sort_values(by="prediccion_mc", ascending=True).reset_index(drop=True)

    # Crear colores personalizados
    colores = ['#FF4136'] * len(especie_df)  # Color rojo para todas las barras
    indice_label = especie_df[especie_df['prediccion_mc'] == label].index
    for idx in indice_label:
        colores[idx] = '#2ECC40'  # Color verde para las barras con label correcto

    # Crear el gráfico de barras
    sns.barplot(
        x=especie_df['prediccion_mc'],
        y=especie_df["proportion"],
        palette=colores,
        edgecolor="black",
        linewidth=0.8,
        ax=ax2
    )

    # Configurar el gráfico de incertidumbres
    ax2.set_title(f'Distribución de predicciones para {titulo_especie}',
                fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Clase predicha', fontsize=10)
    ax2.set_ylabel('Proporción', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Añadir valores sobre las barras
    for i, p in enumerate(ax2.patches):
        height = p.get_height()
        height_100 = height * 100
        if height > 0:
            ax2.text(p.get_x() + p.get_width()/2., height + 0.01,
                    f'{height_100:.1f}%', ha="center", fontsize=9)

    ax2.set_ylim(0, 1)

    # Eliminar líneas del marco excepto abajo y a la izquierda
    sns.despine(ax=ax2)

    plt.tight_layout()
    plt.show()
