# **Bird Songs**

![Tangara multicolor](https://cdn.download.ams.birds.cornell.edu/api/v1/asset/325521391/2400)

## Schema

> - Objetivo
> - Universo
> - Data
> - To Do
> - Roadmap
> - Modelos base
> - Referencias

## **Objetivo general:**

- Clasificar especies de Passeriformes neotropicales de acuerdo a su canto entendiendo sus dinamicas evolutivas

## Universo

![Taxonomia](https://static.todamateria.com.br/upload/hi/er/hierarquiataxonomica-cke.jpg)

- **Calidad del canto**

  > A, B y C

- **Características del canto**

  > 1. Longitud de onda: Se refiere a la longitud total de la señal de audio, medida en tiempo. En el contexto digital, esto puede ser la duración del archivo de audio en segundos.
  > 2. Intensidad mínima (RMS mínima): Es el valor mínimo de la raíz cuadrada media (RMS) de la amplitud de la señal de audio. La RMS proporciona una medida de la potencia o energía de la señal. La intensidad mínima indica el punto más bajo de energía en la señal de audio.
  > 3. Intensidad media (RMS media): Es el valor medio de la RMS de la señal de audio. Esto da una medida promedio de la energía a lo largo de toda la señal, ofreciendo una idea de cuán "fuerte" o "suave" es la señal en promedio.
  > 4. Intensidad máxima (RMS máxima): Es el valor máximo de la RMS de la señal de audio. Esto indica el punto más alto de energía en la señal de audio.
  > 5. Tonos principales: Es el número de tonos dominantes en la señal de audio, a menudo relacionados con las frecuencias fundamentales de los sonidos predominantes en la señal.
  > 6. Melodía: Representa la secuencia de notas musicales en una señal de audio. La melodía promedio puede ser calculada a partir de la frecuencia fundamental predominante en la señal de audio.
  > 7. Centroide espectral: Es el centroide espectral promedio de la señal de audio. Indica el "centro de masa" del espectro de frecuencias, proporcionando una medida de la "brillantez" de la señal.
  > 8. Rolloff espectral: Es la frecuencia por debajo de la cual se encuentra una cierta cantidad de la energía espectral, comúnmente el 85% o 95% de la energía total. Esto ayuda a caracterizar el contenido de alta frecuencia de la señal.
  > 9. Contraste espectral: Es la diferencia entre los picos y los valles en un espectro de frecuencias. Indica la variabilidad de la energía a lo largo de diferentes bandas de frecuencia.
  > 10. Ancho de banda espectral: Es el rango de frecuencias en el cual se encuentra la mayor parte de la energía de la señal de audio. Es una medida de la dispersión de las frecuencias alrededor del centroide espectral.
  > 11. Croma: Es una representación de la distribución de energía por tonos, agrupando las frecuencias en 12 clases que corresponden a las 12 notas de la escala musical.
  > 12. Tempo: Es la velocidad o el ritmo de una pieza musical, usualmente medido en beats per minute (BPM).
  > 13. Pulso: Representa la regularidad y la estructura rítmica de una señal de audio.
  > 14. Coeficientes MFCC (Mel-Frequency Cepstral Coefficients): Son los coeficientes que representan el espectro de potencia de una señal de audio, comprimidos en una escala de frecuencia mel. Se usan ampliamente en el reconocimiento de voz y análisis de audio.
  > 15. RMS (Root Mean Square): Es la raíz cuadrada media de la señal de audio, proporcionando una medida de la energía o potencia de la señal.
  > 16. Cens: Es una versión suavizada del croma, útil para la identificación robusta de la tonalidad.
  > 17. Piptrack: Representa los picos en la señal de audio, usados para identificar las frecuencias prominentes.
  > 18. Cruces por cero: Es la tasa a la que la señal cambia de signo (de positivo a negativo o viceversa), usada para analizar la frecuencia de la señal.
  > 19. Cromagrama CQT: Es una representación del croma utilizando una transformada de calidad constante, proporcionando una representación del contenido armónico.
  > 20. Cromagrama CENS: Es una versión suavizada del cromagrama CQT, utilizado para análisis de patrones tonales.
  > 21. Melspectrogram: Es una representación del espectro de potencia en una escala de frecuencia mel, que es más alineada con la percepción humana del sonido.
  > 22. Polynomial coefficients: Son los coeficientes de un polinomio ajustado a la señal de audio, usados para modelar variaciones en la señal.
  > 23. Fourier tempogram: Es una representación del ritmo basada en la transformada de Fourier, mostrando la periodicidad temporal de la señal.

- **Passeriformes**

## **Data**

- [XenoCanto](https://xeno-canto.org/)

## **To Do**

- Conseguir información extra (colecciones)
- Subir sonidos al repo
- Analizar la muestra
  - Analisis del espectograma
  - Analisis usando librosa
- Filogenia
  - Construir o usar una preexistente
  - Análisis evolutivos
- Modelo de clasificación
  - Realizar primera versión con muestreo
  - Levantar infra para entrenar modelo final
- Crear documento
- Publicar

## **Modelos Base**

- [HuggingFace Models](https://huggingface.co/dima806/bird_sounds_classification)
- [Wavml-large](https://huggingface.co/saadashraf/birds_model)
- [Wav2vec2-base](https://huggingface.co/Saads/bird_classification_model)
- [Whisper-base](https://huggingface.co/openai/whisper-base)

## **Referencias modelos**

- **Papers**:
  - https://arxiv.org/pdf/2404.10420
  - https://arxiv.org/pdf/2403.10380
  - https://arxiv.org/pdf/2312.15824
  - https://arxiv.org/pdf/2303.10757

## **Referencias estudios de canto de aves**

....
