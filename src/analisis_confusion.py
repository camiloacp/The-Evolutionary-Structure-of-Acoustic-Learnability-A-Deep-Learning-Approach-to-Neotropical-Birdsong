"""
Script para crear una tabla de análisis de confusiones entre especies.
Incluye: especie, familia, accuracy, principal confusión, accuracy de la confusión,
número de muestras y función para calcular F1-score.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, Tuple, Optional
import ast
import re


def calcular_f1_score(precision: float, recall: float) -> float:
    """
    Calcula el F1-score a partir de precisión y recall.
    
    Args:
        precision: Valor de precisión (0-1)
        recall: Valor de recall (0-1)
    
    Returns:
        F1-score calculado
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def parsear_lista_numpy(lista_str: str) -> list:
    """
    Parsea una cadena que representa una lista de numpy a una lista de Python.
    
    Args:
        lista_str: Cadena con formato "[np.int64(76), np.int64(82), ...]"
    
    Returns:
        Lista de enteros
    """
    try:
        # Extraer todos los números usando regex
        numeros = re.findall(r'np\.(?:int64|float32)\((\d+)\)', lista_str)
        if numeros:
            return [int(n) for n in numeros]
        
        # Si no funciona, intentar parsear directamente
        lista_str_limpia = lista_str.replace('np.int64(', '').replace('np.float32(', '').replace(')', '')
        lista = ast.literal_eval(lista_str_limpia)
        return [int(x) if isinstance(x, (int, float)) else int(str(x)) for x in lista]
    except Exception as e:
        # Si falla, intentar parsear directamente
        try:
            return ast.literal_eval(lista_str)
        except:
            return []


def obtener_principal_confusion(predicciones_mc: list, clase_real: int) -> Tuple[Optional[int], int]:
    """
    Encuentra la clase más frecuentemente predicha incorrectamente.
    
    Args:
        predicciones_mc: Lista de predicciones de Monte Carlo
        clase_real: Clase real de la especie
    
    Returns:
        Tupla (clase_confusion, frecuencia)
    """
    # Filtrar solo las predicciones incorrectas
    predicciones_incorrectas = [p for p in predicciones_mc if p != clase_real]
    
    if not predicciones_incorrectas:
        return None, 0
    
    # Contar frecuencias
    contador = Counter(predicciones_incorrectas)
    clase_confusion, frecuencia = contador.most_common(1)[0]
    
    return clase_confusion, frecuencia


def crear_mapeo_clases_especies(
    df_incertidumbres: pd.DataFrame,
    df_reporte: pd.DataFrame
) -> Dict[int, str]:
    """
    Crea un mapeo de clases numéricas a nombres de especies.
    
    Args:
        df_incertidumbres: DataFrame de incertidumbres
        df_reporte: DataFrame de reporte con especies ordenadas
    
    Returns:
        Diccionario clase_numérica -> nombre_especie
    """
    mapeo = {}
    
    # El orden del reporte debería corresponder al orden del label encoder
    especies_ordenadas = df_reporte['species'].tolist()
    
    # Crear mapeo basado en el orden del reporte
    for idx, especie in enumerate(especies_ordenadas):
        mapeo[idx] = especie
    
    # También intentar crear mapeo desde el CSV de incertidumbres
    # para verificar consistencia
    especies_unicas = df_incertidumbres.iloc[:, 0].unique()
    
    for especie in especies_unicas:
        if pd.isna(especie):
            continue
        
        filas_especie = df_incertidumbres[df_incertidumbres.iloc[:, 0] == especie]
        if not filas_especie.empty:
            primera_fila = filas_especie.iloc[0]
            try:
                clase_real_str = str(primera_fila['clase_real'])
                clase_real = parsear_lista_numpy(clase_real_str)
                if clase_real:
                    clase_num = clase_real[0]
                    # Solo actualizar si no existe o si coincide
                    if clase_num not in mapeo:
                        mapeo[clase_num] = especie
                    elif mapeo[clase_num] != especie:
                        # Si hay conflicto, preferir el del reporte
                        pass
            except Exception as e:
                pass
    
    return mapeo


def crear_mapeo_genero_familia() -> Dict[str, str]:
    """
    Crea un mapeo de géneros a familias basado en taxonomia.py.
    
    Returns:
        Diccionario género -> familia
    """
    import os
    import sys
    
    # Obtener la ruta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Intentar diferentes métodos de importación
    genus_family = None
    
    # Método 1: Importación relativa
    try:
        from .taxonomia import genus_family
    except ImportError:
        # Método 2: Agregar al path e importar
        try:
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from src.taxonomia import genus_family
        except ImportError:
            # Método 3: Leer el archivo directamente y ejecutar
            try:
                taxonomia_path = os.path.join(current_dir, 'taxonomia.py')
                with open(taxonomia_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    # Ejecutar el código para obtener genus_family
                    namespace = {}
                    exec(code, namespace)
                    genus_family = namespace.get('genus_family', [])
            except Exception as e:
                print(f"Advertencia: No se pudo cargar taxonomia.py: {e}")
                # Retornar diccionario vacío si falla todo
                return {}
    
    if genus_family is None:
        return {}
    
    mapeo = {}
    for item in genus_family:
        partes = item.split('_')
        if len(partes) >= 2:
            familia = partes[0].capitalize()
            genero = partes[1].capitalize()
            mapeo[genero] = familia
    
    return mapeo


def obtener_familia_desde_especie(especie: str, mapeo_genero_familia: Dict[str, str]) -> str:
    """
    Intenta obtener la familia taxonómica desde el nombre de la especie.
    Extrae el género y busca en el mapeo.
    
    Args:
        especie: Nombre científico de la especie (formato "Genero especie")
        mapeo_genero_familia: Diccionario que mapea géneros a familias
    
    Returns:
        Nombre de la familia o "Desconocida"
    """
    # Extraer género (primera palabra)
    partes = especie.split()
    if partes:
        genero = partes[0].capitalize()
        return mapeo_genero_familia.get(genero, 'Desconocida')
    
    return 'Desconocida'


def crear_tabla_confusiones(
    csv_incertidumbres: str,
    csv_reporte: str
) -> pd.DataFrame:
    """
    Crea una tabla de análisis de confusiones entre especies.
    
    Args:
        csv_incertidumbres: Ruta al CSV de incertidumbres
        csv_reporte: Ruta al CSV de reporte con métricas por especie
    
    Returns:
        DataFrame con las columnas: Especie, Familia, Acc. (%), Principal Confusión,
        Acc. Confusión (%), Número de Muestras, F1-Score
    """
    # Cargar datos
    print("Cargando datos de incertidumbres...")
    df_incertidumbres = pd.read_csv(csv_incertidumbres)
    
    # La primera columna sin nombre contiene las especies
    nombre_columna_especie = df_incertidumbres.columns[0]
    
    print("Cargando datos de reporte...")
    df_reporte = pd.read_csv(csv_reporte)
    
    # Crear diccionario de especies a métricas
    reporte_dict = {}
    for _, row in df_reporte.iterrows():
        especie = row['species']
        reporte_dict[especie] = {
            'precision': row['precision'],
            'recall': row['recall'],
            'f1-score': row['f1-score'],
            'support': row['support'],
            'accuracy': row['recall']  # Usamos recall como aproximación de accuracy
        }
    
    # Crear mapeo de clases a especies
    print("Creando mapeo de clases a especies...")
    mapeo_clases_especies = crear_mapeo_clases_especies(df_incertidumbres, df_reporte)
    
    # Crear mapeo de géneros a familias
    print("Creando mapeo de géneros a familias...")
    mapeo_genero_familia = crear_mapeo_genero_familia()
    
    # Procesar cada especie del reporte
    resultados = []
    
    print("Procesando especies...")
    for idx, row in df_reporte.iterrows():
        especie = row['species']
        
        # Obtener métricas del reporte
        metricas = reporte_dict[especie]
        accuracy = metricas['accuracy'] * 100  # Convertir a porcentaje
        num_muestras = metricas['support']
        f1_score = metricas['f1-score']
        
        # Obtener familia
        familia = obtener_familia_desde_especie(especie, mapeo_genero_familia)
        
        # Buscar en el CSV de incertidumbres
        filas_especie = df_incertidumbres[df_incertidumbres[nombre_columna_especie] == especie]
        
        principal_confusion_especie = None
        accuracy_confusion = None
        
        if not filas_especie.empty:
            # Obtener clase real y predicciones de la primera fila
            primera_fila = filas_especie.iloc[0]
            
            try:
                clase_real_str = str(primera_fila['clase_real'])
                predicciones_str = str(primera_fila['predicciones_mc'])
                
                clase_real = parsear_lista_numpy(clase_real_str)
                predicciones_mc = parsear_lista_numpy(predicciones_str)
                
                if clase_real and predicciones_mc:
                    clase_real_val = clase_real[0]
                    
                    # Encontrar principal confusión
                    clase_confusion, _ = obtener_principal_confusion(predicciones_mc, clase_real_val)
                    
                    if clase_confusion is not None and clase_confusion in mapeo_clases_especies:
                        principal_confusion_especie = mapeo_clases_especies[clase_confusion]
                        
                        # Obtener accuracy de la confusión
                        if principal_confusion_especie in reporte_dict:
                            accuracy_confusion = reporte_dict[principal_confusion_especie]['accuracy'] * 100
            except Exception as e:
                print(f"Error procesando {especie}: {e}")
        
        resultados.append({
            'Especie': especie,
            'Familia': familia,
            'Acc. (%)': round(accuracy, 1),
            'Principal Confusión': principal_confusion_especie if principal_confusion_especie else 'N/A',
            'Acc. Confusión (%)': round(accuracy_confusion, 1) if accuracy_confusion is not None else 'N/A',
            'Número de Muestras': int(num_muestras),
            'F1-Score': round(f1_score, 3)
        })
    
    df_resultado = pd.DataFrame(resultados)
    
    # Ordenar por accuracy descendente
    df_resultado = df_resultado.sort_values('Acc. (%)', ascending=False)
    
    # Reemplazar NaN con 'N/A' para consistencia
    df_resultado['Principal Confusión'] = df_resultado['Principal Confusión'].fillna('N/A')
    df_resultado['Acc. Confusión (%)'] = df_resultado['Acc. Confusión (%)'].replace('N/A', 'N/A')
    
    return df_resultado


if __name__ == "__main__":
    # Rutas a los archivos
    csv_incertidumbres = "src/data/incertidumbres_ResNet152V2.csv"
    csv_reporte = "src/data/reporte_resnet.csv"
    
    # Crear tabla
    df_tabla = crear_tabla_confusiones(csv_incertidumbres, csv_reporte)
    
    # Mostrar resultados
    print("\n" + "="*100)
    print("TABLA DE ANÁLISIS DE CONFUSIONES")
    print("="*100)
    print(df_tabla.to_string(index=False))
    
    # Guardar a CSV (reemplazar NaN con N/A antes de guardar)
    output_file = "src/data/tabla_confusiones_analisis.csv"
    df_tabla_guardar = df_tabla.copy()
    df_tabla_guardar['Principal Confusión'] = df_tabla_guardar['Principal Confusión'].replace('N/A', 'N/A')
    df_tabla_guardar['Acc. Confusión (%)'] = df_tabla_guardar['Acc. Confusión (%)'].replace('N/A', 'N/A')
    df_tabla_guardar.to_csv(output_file, index=False, na_rep='N/A')
    print(f"\nTabla guardada en: {output_file}")
    
    # Mostrar estadísticas
    print(f"\nTotal de especies analizadas: {len(df_tabla)}")
    print(f"Especies con confusión identificada: {len(df_tabla[df_tabla['Principal Confusión'] != 'N/A'])}")
    print(f"Accuracy promedio: {df_tabla['Acc. (%)'].mean():.1f}%")
    print(f"F1-Score promedio: {df_tabla['F1-Score'].mean():.3f}")
