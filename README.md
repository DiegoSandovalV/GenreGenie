# GenreGenie

## Descripcción

GenreGenie es un clasificador de géneros musicales basado en el aprendizaje automático. Utiliza un modelo de red neuronal para predecir el género de una canción a partir de sus características de audio.

## DataSet

El dataset utilizado para entrenar el modelo es el [GTZAN Genre Collection](https://www.kaggle.com/code/dapy15/music-genre-classification#Classifier), que contiene 1000 canciones de 10 géneros diferentes. Cada género tiene 100 canciones de 30 segundos de duración.

Los géneros son:

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

El data set fue utilizado para generar 10 [espectrogramas](#espectogramas) de 3 segundos de cada canción. Por lo tanto, el dataset contiene 1000 espectrogramas de cada género, lo que da un total de 10,000 espectrogramas.

# Espectogramas

Los espectrogramas son representaciones visuales de la frecuencia y la amplitud de una señal de audio a lo largo del tiempo. En este proyecto, se utilizan espectrogramas para representar las características de audio de las canciones y entrenar el modelo de red neuronal.

### Espectogramas Mel

Los espectrogramas Mel son una representación de la frecuencia de una señal de audio en la escala Mel, que es una escala logarítmica que se asemeja a la percepción humana del sonido. Esta representación es útil para el reconocimiento de patrones en señales de audio.

Se emplea la librería [librosa](https://librosa.org/doc/main/index.html) para generar los espectrogramas Mel. La función `librosa.feature.melspectrogram` se utiliza para calcular el espectrograma Mel a partir del audio.

Los parámetros utilizados para generar los espectrogramas Mel son:

- `SampleRate`: 22050 Hz

Define cuantos datos se toman por segundo. 22050 Hz es un estándar común para capturar los detalles del audio sin incluir el ruido de alta frecuencia.

- `n_mels`: 128

Define el número de bandas Mel. Las cuales se utilizan para representar la frecuencia de la señal de audio. 128 es un valor comúnmente utilizado que proporciona un buen equilibrio entre la resolución temporal y la frecuencia.

- `Duracion del segmento`: 3 segundos


 En el artículo "Music Genre Classification: A Comparative Analysis of CNN and XGBoost Approaches..." (Meng, Y.) se utilizaron segmentos de igual duración para la extracción de características MFCC y en modelos como CNN y XGBoost. Se observó que la segmentación de datos puede mejorar el rendimiento, especialmente en las CNN.

## División del dataset

El dataset se divide en un 60% para entrenamiento, 20% para validación y 20% para prueba, manteniendo uniformidad entre los géneros. Esta división se realizó de manera aleatoria al momento de generar los espectrogramas.

## Referencias

[1]Y. Meng, “Music Genre Classification: A Comparative Analysis of CNN and XGBoost Approaches with Mel-frequency cepstral coefficients and Mel Spectrograms,” arXiv (Cornell University), Jan. 2024, doi: https://doi.org/10.48550/arxiv.2401.04737.
