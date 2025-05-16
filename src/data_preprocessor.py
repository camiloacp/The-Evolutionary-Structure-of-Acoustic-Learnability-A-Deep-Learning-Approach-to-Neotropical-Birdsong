import pandas as pd
import logging
from typing import Optional, List
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection, ProbeFeatureSelection
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass, field


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass(slots=True)
class DataPreprocessor:
    """
    Clase para preprocesamiento de datos.
    Mantiene valores por defecto para umbrales, estrategias y parámetros
    que pueden ajustarse en la inicialización.

    Parámetros:
        null_threshold (float): Umbral para eliminar columnas con demasiados valores nulos (0-1).
            Por defecto 0.9.
        constant_threshold (float): Umbral para eliminar columnas con valores constantes (0-1).
            Por defecto 1.0.
        corr_threshold (float): Umbral para detectar y eliminar correlaciones entre variables (0-1).
            Por defecto 0.85.
        corr_method (str): Método para calcular correlaciones ('pearson', 'spearman', etc).
            Por defecto 'spearman'.
        corr_strategy (Optional[str]): Estrategia de preprocesado para correlación.
            Opciones: 'normalizar', 'estandarizar' o None. Por defecto None.
        estimator (object): Estimador para selección de características.
            Por defecto RandomForestClassifier.
        scoring (str): Métrica de evaluación para selección de características.
            Por defecto 'roc_auc'.
        cv (int): Número de folds para validación cruzada.
            Por defecto 3.
        feature_selection_strategy (str): Estrategia para selección de características.
            Por defecto 'probe_feature_selection'.
        target_column (str): Nombre de la columna objetivo.
            Por defecto 'target'.
    """
    # Atributos relacionados con el preprocesado de nulos y valores constantes
    null_threshold: float = 0.95
    constant_threshold: float = 0.9

    # Atributos para la limpieza de correlación
    corr_threshold: float = 0.85
    corr_method: str = "spearman"
    corr_strategy: Optional[str] = None  # "normalizar", "estandarizar" o None

    # Atributos para la selección de características
    estimator: object = field(default_factory=lambda: RandomForestClassifier(random_state=42))
    scoring: str = "roc_auc"
    cv: int = 3
    feature_selection_strategy: str = "probe_feature_selection"

    # Columna objetivo por defecto
    target_column: str = "target"

    def _validar_threshold(self, threshold: float) -> None:
        """Valida que el threshold esté entre 0 y 1."""
        if not 0 <= threshold <= 1:
            raise ValueError("El threshold debe estar entre 0 y 1")

    def cast_to_numeric(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convierte las columnas del DataFrame a tipo numérico.

        Args:
            df: DataFrame a convertir
            exclude_columns: Lista de columnas a excluir del procesamiento

        Returns:
            DataFrame con columnas convertidas a numéricas cuando es posible
        """
        df_copy = df.copy()

        # Columnas a procesar (todas excepto las excluidas)
        columns_to_process = df_copy.columns if exclude_columns is None else [col for col in df_copy.columns if col not in exclude_columns]

        for col in columns_to_process:
            try:
                df_copy[col] = pd.to_numeric(df_copy[col])
            except (ValueError, TypeError):
                pass
        return df_copy

    def normalizar(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normaliza las columnas numéricas del DataFrame.

        Args:
            df (pd.DataFrame): DataFrame a normalizar
            exclude_columns: Lista de columnas a excluir del procesamiento

        Returns:
            pd.DataFrame: DataFrame normalizado
        """
        df_normalizado = self.cast_to_numeric(df.copy(), exclude_columns)

        # Determinar columnas a procesar (excluir las especificadas)
        columns_to_process = df_normalizado.columns if exclude_columns is None else [col for col in df_normalizado.columns if col not in exclude_columns]

        # Filtrar columnas numéricas y aplicar normalización vectorizada
        numeric_cols = df_normalizado[columns_to_process].select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            min_val = df_normalizado[col].min()
            max_val = df_normalizado[col].max()
            if max_val > min_val:  # Evitar división por cero
                df_normalizado[col] = (df_normalizado[col] - min_val) / (max_val - min_val)

        return df_normalizado

    def estandarizar(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Estandariza las columnas numéricas del DataFrame.

        Args:
            df (pd.DataFrame): DataFrame a estandarizar
            exclude_columns: Lista de columnas a excluir del procesamiento

        Returns:
            pd.DataFrame: DataFrame estandarizado
        """
        df_estandarizado = self.cast_to_numeric(df.copy(), exclude_columns)

        # Determinar columnas a procesar (excluir las especificadas)
        columns_to_process = df_estandarizado.columns if exclude_columns is None else [col for col in df_estandarizado.columns if col not in exclude_columns]

        # Filtrar columnas numéricas y aplicar estandarización vectorizada
        numeric_cols = df_estandarizado[columns_to_process].select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            mean_val = df_estandarizado[col].mean()
            std_val = df_estandarizado[col].std()
            if std_val > 0:  # Evitar división por cero
                df_estandarizado[col] = (df_estandarizado[col] - mean_val) / std_val

        return df_estandarizado

    def limpiar_nulos(self, df: pd.DataFrame, threshold: Optional[float]=None, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Elimina las columnas del DataFrame que tienen un porcentaje de valores nulos mayor al límite especificado.

        Args:
            df (pandas.DataFrame): DataFrame a procesar
            threshold (float): Límite de valores nulos permitidos (entre 0 y 1)
            exclude_columns: Lista de columnas a excluir del procesamiento

        Returns:
            pandas.DataFrame: DataFrame sin las columnas que exceden el límite de nulos
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("El argumento df debe ser un DataFrame de pandas")

        if threshold is None:
            threshold = self.null_threshold

        self._validar_threshold(threshold)

        # Separar columnas excluidas y a procesar
        df_excluidas = pd.DataFrame()
        if exclude_columns:
            excluded_cols = [col for col in exclude_columns if col in df.columns]
            df_excluidas = df[excluded_cols]
            df_a_procesar = df.drop(columns=excluded_cols, errors='ignore')
        else:
            df_a_procesar = df.copy()

        # Eliminar columnas con demasiados nulos en las columnas a procesar
        min_non_nulls = int(len(df_a_procesar) * (1 - threshold))
        df_sin_nulos = df_a_procesar.dropna(axis=1, thresh=min_non_nulls)

        logging.info(f"Columnas con nulos eliminadas: {df_a_procesar.shape[1] - df_sin_nulos.shape[1]}")
        logging.info(f"Forma del dataframe tras eliminar columnas con nulos: {df_sin_nulos.shape}")

        # Reunir las columnas excluidas con el resultado procesado
        if not df_excluidas.empty:
            return pd.concat([df_sin_nulos, df_excluidas], axis=1)
        return df_sin_nulos

    def limpiar_duplicados(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Elimina las filas duplicadas del DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame a procesar
            exclude_columns: Lista de columnas a excluir al verificar duplicados

        Returns:
            pandas.DataFrame: DataFrame sin filas duplicadas
        """
        # Determinar columnas a considerar para duplicados
        columns_to_consider = df.columns if exclude_columns is None else [col for col in df.columns if col not in exclude_columns]

        # Eliminar duplicados considerando solo las columnas especificadas
        if len(columns_to_consider) < len(df.columns):
            df_sin_duplicados = df.drop_duplicates(subset=columns_to_consider)
        else:
            df_sin_duplicados = df.drop_duplicates()

        logging.info(f"Filas con duplicados: {df.shape[0] - df_sin_duplicados.shape[0]}")
        logging.info(f"Forma tras eliminar filas duplicadas: {df_sin_duplicados.shape}")
        return df_sin_duplicados

    def limpiar_constantes(self, df: pd.DataFrame, threshold: Optional[float]=None, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Elimina las columnas del DataFrame que tienen valores casi constantes.

        Nota: Se recomienda usar este método con un dataframe normalizado o estandarizado.

        Args:
            df (pandas.DataFrame): DataFrame a procesar
            threshold (float): Umbral de tolerancia para considerar una columna como constante (entre 0 y 1)
            exclude_columns: Lista de columnas a excluir del procesamiento

        Returns:
            pandas.DataFrame: DataFrame sin las columnas casi constantes
        """
        if threshold is None:
            threshold = self.constant_threshold

        self._validar_threshold(threshold)

        # Separar columnas excluidas y a procesar
        df_excluidas = pd.DataFrame()
        if exclude_columns:
            excluded_cols = [col for col in exclude_columns if col in df.columns]
            df_excluidas = df[excluded_cols]
            df_a_procesar = df.drop(columns=excluded_cols, errors='ignore')
        else:
            df_a_procesar = df.copy()

        transformer = DropConstantFeatures(tol=threshold, missing_values='ignore')
        df_no_const = transformer.fit_transform(df_a_procesar)

        logging.info(f"Columnas constantes eliminadas: {df_a_procesar.shape[1] - df_no_const.shape[1]}")
        logging.info(f"Forma tras eliminar columnas constantes: {df_no_const.shape}")

        # Reunir las columnas excluidas con el resultado procesado
        if not df_excluidas.empty:
            return pd.concat([df_no_const, df_excluidas], axis=1)
        return df_no_const

    def limpiar_correlacion(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        method: Optional[str] = None,
        strategy: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Elimina las columnas del DataFrame que tienen una alta correlación entre sí.

        Args:
            df (pandas.DataFrame): DataFrame a procesar
            threshold (float): Umbral de correlación para eliminar columnas (entre 0 y 1)
            method (str): Método de correlación a utilizar ('spearman' o 'pearson') por defecto es spearman
            strategy (str): Estrategia de normalización a utilizar ('normalizar' o 'estandarizar') por defecto es None
            exclude_columns: Lista de columnas a excluir del procesamiento

        Returns:
            pandas.DataFrame: DataFrame sin las columnas altamente correlacionadas

        Nota: Si se utiliza la estrategia de normalización, se recomienda utilizar el método de Spearman para calcular la correlación.
        """
        if threshold is None:
            threshold = self.corr_threshold

        self._validar_threshold(threshold)

        if method is None:
            method = self.corr_method

        if strategy is None:
            strategy = self.corr_strategy

        # Separar columnas excluidas y a procesar
        df_excluidas = pd.DataFrame()
        if exclude_columns:
            excluded_cols = [col for col in exclude_columns if col in df.columns]
            df_excluidas = df[excluded_cols]
            df_a_procesar = df.drop(columns=excluded_cols, errors='ignore')
        else:
            df_a_procesar = df.copy()

        # Aplicar la estrategia de preparación de datos seleccionada
        if strategy == 'normalizar':
            df_procesado = self.normalizar(df_a_procesar)
        elif strategy == 'estandarizar':
            df_procesado = self.estandarizar(df_a_procesar)
        else:
            df_procesado = df_a_procesar.copy()

        # Aplicar selección de características correlacionadas
        transformer = SmartCorrelatedSelection(
            method=method,
            threshold=threshold,
            missing_values='ignore'
        )
        df_no_corr = transformer.fit_transform(df_procesado)

        logging.info(f"Columnas correlacionadas eliminadas: {df_a_procesar.shape[1] - df_no_corr.shape[1]}")
        logging.info(f"Forma tras eliminar columnas correlacionadas: {df_no_corr.shape}")

        # Devolver el dataframe original filtrado con las columnas seleccionadas más las excluidas
        resultado = df_a_procesar[df_no_corr.columns]
        if not df_excluidas.empty:
            return pd.concat([resultado, df_excluidas], axis=1)
        return resultado

    def model_feature_selection(
        self,
        df: pd.DataFrame,
        estimator: Optional[object] = None,
        scoring: Optional[str] = None,
        cv: Optional[int] = None,
        strategy: Optional[str] = None,
        target_column: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Realiza selección de características utilizando diferentes estrategias basadas en modelos.

        Args:
            df (pandas.DataFrame): DataFrame a procesar
            estimator: Estimador a utilizar para la selección de características. Por defecto RandomForestClassifier
            scoring (str): Métrica de evaluación a utilizar. Por defecto 'roc_auc'
            cv (int): Número de folds para validación cruzada. Por defecto 3
            strategy (str): Estrategia de selección a utilizar ('probe_feature_selection' o 'boruta_shap').
                          Por defecto 'probe_feature_selection'
            target_column (str): Nombre de la columna objetivo. Por defecto 'target'
            exclude_columns: Lista de columnas a excluir del procesamiento

        Returns:
            pandas.DataFrame: DataFrame con las características seleccionadas

        Nota:
            - La estrategia 'probe_feature_selection' utiliza ProbeFeatureSelection para seleccionar características
            - La estrategia 'boruta_shap' utiliza BorutaShap y excluye automáticamente las variables categóricas y debe usarse con valores nulos imputados.
        """
        if estimator is None:
            estimator = self.estimator
        if scoring is None:
            scoring = self.scoring
        if cv is None:
            cv = self.cv
        if strategy is None:
            strategy = self.feature_selection_strategy
        if target_column is None:
            target_column = self.target_column

        if target_column not in df.columns:
            raise ValueError(f"La columna objetivo '{target_column}' no existe en el DataFrame")

        # Gestionar columnas excluidas
        if exclude_columns is None:
            exclude_columns = []

        # Asegurarse que target_column no esté en exclude_columns para evitar problemas
        if target_column in exclude_columns:
            exclude_columns = [col for col in exclude_columns if col != target_column]

        # Columnas a considerar para el procesamiento
        columns_to_process = [col for col in df.columns if col != target_column and col not in exclude_columns]

        X = df[columns_to_process]
        y = df[target_column]

        # Mantener las columnas excluidas aparte
        excluded_data = df[exclude_columns] if exclude_columns else pd.DataFrame()

        if strategy == "probe_feature_selection":
            logging.info("Estrategia de selección: ProbeFeatureSelection")

            transformer = ProbeFeatureSelection(
                estimator=estimator,
                scoring=scoring,
                n_probes=3,
                distribution="all",
                cv=cv,
                random_state=42
            )
            X_selected = transformer.fit_transform(X, y)

            logging.info(f"Columnas eliminadas con ProbeFeatureSelection: {X.shape[1] - X_selected.shape[1]}")
            logging.info(f"Forma tras eliminar columnas con ProbeFeatureSelection: {X_selected.shape}")

            # Combinar las columnas seleccionadas, la columna objetivo y las excluidas
            selected_cols = list(X_selected.columns)
            final_cols = selected_cols + [target_column] + list(exclude_columns)
            return df[final_cols]

        else:
            logging.warning(f"Estrategia de selección '{strategy}' no válida. Se devuelve el DataFrame original.")
            return df

    def procesar_pipeline(
        self,
        df: pd.DataFrame,
        ejecutar_limpieza_nulos: bool = True,
        ejecutar_limpieza_duplicados: bool = True,
        ejecutar_limpieza_constantes: bool = True,
        ejecutar_limpieza_correlacion: bool = True,
        ejecutar_seleccion_caracteristicas: bool = False,
        target_column: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Ejecuta un pipeline completo de preprocesamiento.

        Args:
            df (pd.DataFrame): DataFrame a procesar
            ejecutar_limpieza_nulos (bool): Si se debe eliminar columnas con muchos nulos
            ejecutar_limpieza_duplicados (bool): Si se debe eliminar filas duplicadas
            ejecutar_limpieza_constantes (bool): Si se debe eliminar columnas constantes
            ejecutar_limpieza_correlacion (bool): Si se debe eliminar columnas correlacionadas
            ejecutar_seleccion_caracteristicas (bool): Si se debe realizar selección de características
            target_column (str): Nombre de la columna objetivo
            exclude_columns: Lista de columnas a excluir de todos los procesamientos

        Returns:
            pd.DataFrame: DataFrame procesado
        """
        logging.info("Iniciando pipeline de preprocesamiento")

        df = df.copy()
        if target_column is None:
            target_column = self.target_column

        if ejecutar_limpieza_nulos:
            df = self.limpiar_nulos(df, exclude_columns=exclude_columns)

        if ejecutar_limpieza_duplicados:
            df = self.limpiar_duplicados(df, exclude_columns=exclude_columns)

        if ejecutar_limpieza_constantes:
            df = self.limpiar_constantes(df, exclude_columns=exclude_columns)

        if ejecutar_limpieza_correlacion:
            df = self.limpiar_correlacion(df, exclude_columns=exclude_columns)

        if ejecutar_seleccion_caracteristicas:
            df = self.model_feature_selection(df, target_column=target_column, exclude_columns=exclude_columns)

        logging.info(f"Pipeline de preprocesamiento completado. Forma final: {df.shape}")
        return df
