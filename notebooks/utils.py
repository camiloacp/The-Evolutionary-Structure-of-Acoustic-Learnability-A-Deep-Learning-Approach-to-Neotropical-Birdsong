import pandas as pd

def sinonimias(df: pd.DataFrame, avonet1: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces specific species names in the 'Species1' column of the 'avonet1' DataFrame 
    and the 'Specie' column of the 'df' DataFrame with their corresponding synonyms.
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the 'Specie' column.
    - avonet1 (pd.DataFrame): The DataFrame containing the 'Species1' column.
    Returns:
    - pd.DataFrame: The modified 'df' DataFrame with replaced species names in the 'Specie' column.
    - pd.DataFrame: The modified 'avonet1' DataFrame with replaced species names in the 'Species1' column.
    """
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_vitriolina", "Stilpnia_vitriolina")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_nigrocincta", "Stilpnia_nigrocincta")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Spodiornis_rusticus", "Haplospiza_rustica")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Vireo_pallens", "Vireo_approximans")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_palmarum", "Thraupis_palmarum")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_glaucocolpa", "Thraupis_glaucocolpa")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_episcopus", "Thraupis_episcopus")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Maschalethraupis_surinama", "Tachyphonus_surinamus")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Chrysocorypha_delatrii", "Tachyphonus_delatrii")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_larvata", "Stilpnia_larvata")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_heinei", "Stilpnia_heinei")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_cyanoptera", "Stilpnia_cyanoptera")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_cyanicollis", "Stilpnia_cyanicollis")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_cayana", "Stilpnia_cayana")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Pyriglena_leuconota", "Pyriglena_maura")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Pseudocolaptes_boissonneauii", "Pseudocolaptes_boissonneautii")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Polioptila_guianensis", "Polioptila_facilis")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Islerothraupis_cristata", "Loriotus_cristatus")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Grallaria_fenwickorum", "Grallaria_urraoensis")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Islerothraupis_luctuosa", "Loriotus_luctuosus")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_ruficervix", "Chalcothraupis_ruficervix")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Euphonia_cyanocephala", "Chlorophonia_cyanocephala")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Thripophaga_gutturata", "Cranioleuca_gutturata")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Philydor_erythropterum", "Dendroma_erythroptera")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Philydor_rufum", "Dendroma_rufa")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_guttata", "Ixothraupis_guttata")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_punctata", "Ixothraupis_punctata")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_rufigula", "Ixothraupis_rufigula")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_varia", "Ixothraupis_varia")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Tangara_xanthogastra", "Ixothraupis_xanthogastra")
    avonet1['Species1'] = avonet1['Species1'].str.replace("Chloropipo_flavicapilla", "Xenopipo_flavicapilla")
    
    df['Specie'] = df['Specie'].str.replace("Uromyias_agilis", "Anairetes_agilis")
    df['Specie'] = df['Specie'].str.replace("Xenops_rutilans", "Xenops_rutilus")
    df['Specie'] = df['Specie'].str.replace("Chloropipo_flavicapilla", "Xenopipo_flavicapilla")
    df['Specie'] = df['Specie'].str.replace("Premnornis_guttuliger", "Premnornis_guttuligera")
    #df['Specie'] = df['Specie'].str.replace("Gymnopithys_bicolor", "Gymnopithys_leucaspis")
    
    return df, avonet1

def elimina_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina las columnas de un dataframe que tienen más de 99% valores nulos y retorna el dataframe modificado.
    
    Parameters:
    - df (pd.DataFrame): El DataFrame a procesar.
    
    Returns:
    - pd.DataFrame: El DataFrame modificado sin las columnas con más de 99% valores nulos.
    """
    threshold = 0.99
    # Identificar las columnas con más de 99% valores nulos
    cols_to_drop = df.columns[df.isnull().mean() > threshold]
    # Eliminar las columnas identificadas
    df = df.drop(columns=cols_to_drop)
    return df

def remove_collinear_features(x, threshold):
    '''
    Objetivo:
        Eliminar características colineales en un marco de datos con un coeficiente de correlación
        mayor que el umbral. La eliminación de características colineales puede ayudar a un modelo
        generalizar y mejora la interpretabilidad del modelo.
        
    Entradas:
        x: marco de datos de características
        umbral: se eliminan las entidades con correlaciones superiores a este valor
        
    Producción:
        marco de datos que contiene solo las características no altamente colineales
    '''
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    
    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])
                
    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    return x