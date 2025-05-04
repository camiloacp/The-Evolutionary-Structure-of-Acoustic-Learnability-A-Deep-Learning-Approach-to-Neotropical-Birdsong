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

def calcular_incertidumbre(dataset_espectrogramas, modelo, visualizar=False, ensemble_size=2000):
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
            probabilidades_inferencia = tf.nn.softmax(logits_inferencia, axis=-1).numpy()
            clase_predicha_inferencia = np.argmax(probabilidades_inferencia, axis=-1)[0]
            etiqueta_inferencia_lista.append(clase_predicha_inferencia)

            # Para almacenar todas las probabilidades para los intervalos de confianza
            predicted_probabilities = np.zeros((ensemble_size, probabilidades_inferencia.shape[1]))

            # MONTE CARLO DROPOUT: Múltiples pasadas con training=True
            for i in range(ensemble_size):
                logits = modelo(imagen_batch, training=True)
                probabilidades = tf.nn.softmax(logits, axis=-1).numpy()
                predicted_probabilities[i] = probabilidades[0]  # Guardamos todas las probabilidades
                probabilidades_todas.append(probabilidades[0])
                clase_predicha = np.argmax(probabilidades, axis=-1)[0]
                confianza = probabilidades[0, clase_predicha]

                predicciones_mc.append(clase_predicha)
                confianzas_mc.append(confianza)

            # Visualizar los intervalos de confianza si se solicita
            if visualizar:
                # Calcular la media de las probabilidades para todas las clases
                prob_media = np.mean(predicted_probabilities, axis=0)

                # Obtener los índices del top 10 de las clases con mayor probabilidad media
                top_indices = np.argsort(prob_media)[-10:][::-1]

                # Asegurar que la clase real esté incluida en el conjunto de clases a visualizar
                if etiqueta_real.numpy() not in top_indices:
                    top_indices = np.append(top_indices, etiqueta_real.numpy())

                # Crear la figura con dos subplots
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4),
                                       gridspec_kw={'width_ratios': [2, 4]})

                # Mostrar la imagen y la etiqueta verdadera
                if len(imagen.shape) == 3 and imagen.shape[2] == 3:  # Imagen RGB
                    ax1.imshow(imagen.numpy().astype(np.uint8))
                else:  # Espectrograma u otra imagen de un canal
                    ax1.imshow(imagen.numpy()[..., 0], cmap='gray')
                ax1.axis('off')
                ax1.set_title(f'Etiqueta real: {etiqueta_real.numpy()}')

                # Preparar datos solo para las clases seleccionadas
                pct_2p5 = np.array([np.percentile(predicted_probabilities[:, i], 2.5) for i in top_indices])
                pct_97p5 = np.array([np.percentile(predicted_probabilities[:, i], 97.5) for i in top_indices])

                # Crear barras con colores personalizados
                bar = ax2.bar(range(len(top_indices)), pct_97p5, color='red')

                # Colorear la barra de la clase real en verde
                for i, idx in enumerate(top_indices):
                    if idx == etiqueta_real.numpy():
                        bar[i].set_color('green')
                        break

                # Mostrar el intervalo de confianza inferior
                ax2.bar(range(len(top_indices)), pct_2p5-0.02, color='white', linewidth=1, edgecolor='white')

                # Configurar etiquetas del eje X con los índices de las clases
                ax2.set_xticks(range(len(top_indices)))
                ax2.set_xticklabels(top_indices, rotation=45)

                ax2.set_ylim([0, 1])
                ax2.set_ylabel('Probabilidad')
                ax2.set_title('Top 10 probabilidades con intervalos de confianza del 95%')
                plt.tight_layout()
                plt.show()

    # Calcular la entropía si hay datos disponibles
    entropia = 0
    if probabilidades_todas:
        # Calculamos la entropía promedio
        probabilidades_array = np.array(probabilidades_todas)
        prob_media = np.mean(probabilidades_array, axis=0)
        entropia = -np.sum(prob_media * np.log(prob_media + 1e-9))

    return predicciones_mc, confianzas_mc, etiqueta_inferencia_lista, etiqueta_real_lista, entropia, predicted_probabilities #if dataset_espectrogramas else None

def crear_dataframe_incertidumbre(data, muestra, modelo, preprocessor):
    # Generar dataset de especies
    dataset_generado = especie_batch(muestra)

    # Recolectar resultados de cada especie
    resultados = {}
    for i in range(data["label"].nunique()):
        especie, dataset_especie = specie_ds(data, dataset_generado, preprocessor)
        if especie is None:
            break

        # Modificación aquí: capturar el sexto valor (predicted_probabilities)
        predicciones_mc, confianzas_mc, clase_predicha_inferencia, etiqueta_real, entropia, predicted_probabilities = calcular_incertidumbre(dataset_especie, modelo, visualizar=False)
        resultados[especie] = {
            "clase_real": etiqueta_real,
            "clase_predicha_inferencia": clase_predicha_inferencia,
            "predicciones_mc": predicciones_mc,
            "confianzas_mc": confianzas_mc,
            "entropia": entropia,
            "predicted_probabilities": predicted_probabilities
        }

    # El resto de la función permanece igual
    filas = []
    for especie, datos in resultados.items():
        clase_real = datos["clase_real"]
        clase_predicha = datos["clase_predicha_inferencia"]
        predicciones = datos["predicciones_mc"]
        confianzas = datos["confianzas_mc"]
        entropia = datos["entropia"]

        for i in range(len(predicciones)):
            etiqueta_real = clase_real[0] if len(clase_real) > 0 else None
            etiqueta_pred = clase_predicha[0] if len(clase_predicha) > 0 else None

            filas.append({
                "especie": especie,
                "clase_real": etiqueta_real,
                "clase_predicha": etiqueta_pred,
                "prediccion_mc": predicciones[i],
                "confianza_mc": confianzas[i] if i < len(confianzas) else None,
                'true_positive': 1 if etiqueta_real == etiqueta_pred else 0,
                'true_positivo_mc': 1 if etiqueta_real == predicciones[i] else 0,
                "entropia": entropia
            })

    return pd.DataFrame(filas), resultados

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

def plot_combinado(ruta_imagen, especie_df, label, titulo_especie, predicted_probabilities=None, etiqueta_real=None, figsize=(15, 5)):
    # Crear figura con tres subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize,
                                        gridspec_kw={'width_ratios': [2, 3, 4]})

    # Primer subplot: Mostrar imagen
    try:
        imagen = cargar_imagen_redimensionada(ruta_imagen)
        ax1.imshow(imagen.numpy().astype(np.uint8))
        ax1.set_title(f"Espectrograma: {titulo_especie}")
        ax1.axis('off')
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        ax1.text(0.5, 0.5, "Error al cargar la imagen",
                 ha='center', va='center')
        ax1.axis('off')

    # Segundo subplot: Gráfico de incertidumbres (distribución)
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
    ax2.set_title(f'Distribución de predicciones',
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

    # Tercer subplot: Top 10 probabilidades con intervalos de confianza (si se proporciona predicted_probabilities)
    if predicted_probabilities is not None and etiqueta_real is not None:
        # Calcular la media de las probabilidades para todas las clases
        prob_media = np.mean(predicted_probabilities, axis=0)

        # Obtener los índices del top 10 de las clases con mayor probabilidad media
        top_indices = np.argsort(prob_media)[-10:][::-1]

        # Asegurar que la clase real esté incluida en el conjunto de clases a visualizar
        if etiqueta_real not in top_indices:
            top_indices = np.append(top_indices, etiqueta_real)

        # Preparar datos solo para las clases seleccionadas
        pct_2p5 = np.array([np.percentile(predicted_probabilities[:, i], 2.5) for i in top_indices])
        pct_97p5 = np.array([np.percentile(predicted_probabilities[:, i], 97.5) for i in top_indices])

        # Crear colores personalizados para las barras - todos rojos por defecto
        colores_prob = ['#FF4136'] * len(top_indices)

        # Encontrar la posición de la etiqueta real y cambiar a verde
        for i, idx in enumerate(top_indices):
            if idx == etiqueta_real:
                colores_prob[i] = '#2ECC40'  # Verde para la clase real
                break

        # Crear barras con colores personalizados
        bars = ax3.bar(range(len(top_indices)), pct_97p5,
                      color=colores_prob,
                      edgecolor='black',
                      linewidth=0.8)

        # Mostrar el intervalo de confianza inferior
        ax3.bar(range(len(top_indices)), pct_2p5-0.02,
               color='white',
               linewidth=1,
               edgecolor='white')

        # Añadir valores de porcentaje sobre las barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            height_100 = height * 100
            if height > 0.01:  # Solo mostrar porcentajes significativos
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height_100:.1f}%', ha="center", fontsize=9)

        # Configurar etiquetas del eje X con los índices de las clases
        ax3.set_xticks(range(len(top_indices)))
        ax3.set_xticklabels(top_indices, rotation=45, ha='right')

        # Mejoras estéticas adicionales
        ax3.set_ylim([0, 1])
        ax3.set_ylabel('Probabilidad')
        ax3.set_title('Top 10 probabilidades con IC 95%',
                     fontsize=12, fontweight='bold', pad=10)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)

        # Añadir un texto explicativo sobre los intervalos de confianza
        ax3.text(0.5, -0.15, "Las barras muestran el intervalo de confianza del 95%",
                ha='center', va='center', transform=ax3.transAxes, fontsize=8, style='italic')
    else:
        ax3.set_title("Datos de probabilidades no disponibles")
        ax3.axis('off')

    # Eliminar líneas del marco excepto abajo y a la izquierda
    sns.despine(ax=ax2)
    if predicted_probabilities is not None:
        sns.despine(ax=ax3)

    plt.tight_layout()
    plt.show()
