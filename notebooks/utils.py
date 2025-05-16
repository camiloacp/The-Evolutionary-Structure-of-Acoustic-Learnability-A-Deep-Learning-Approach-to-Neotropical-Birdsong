import pandas as pd

def elimina_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas con más de 99% valores nulos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a procesar.

    Returns
    -------
    pd.DataFrame
        DataFrame sin las columnas con exceso de valores nulos.
    """
    threshold = 0.99
    # Identificar columnas con más de 99% valores nulos
    porcentaje_nulos = df.isnull().mean()
    cols_to_drop = df.columns[porcentaje_nulos > threshold]

    # Eliminar columnas identificadas
    return df.drop(columns=cols_to_drop)

def remove_collinear_features(dataframe: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Elimina características colineales en un DataFrame.

    Las características con coeficiente de correlación mayor que el umbral
    son eliminadas para mejorar la generalización e interpretabilidad del modelo.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame de características.
    threshold : float
        Umbral de correlación para eliminar características.

    Returns
    -------
    pd.DataFrame
        DataFrame que contiene solo las características no altamente colineales.
    """
    # Calcular matriz de correlación
    corr_matrix = dataframe.corr()
    columnas = corr_matrix.columns
    drop_cols = []

    # Iterar por la matriz de correlación comparando correlaciones
    for i in range(len(columnas) - 1):
        for j in range(i + 1):
            # Obtener el valor de correlación entre la columna i+1 y j
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns[0]
            row = item.index[0]
            val = abs(item.values[0][0])

            # Si la correlación excede el umbral
            if val >= threshold:
                # Imprimir características correlacionadas y valor
                print(f"{col} | {row} | {val:.2f}")
                drop_cols.append(col)

    # Eliminar una de cada par de columnas correlacionadas
    drops = set(drop_cols)
    return dataframe.drop(columns=drops)
