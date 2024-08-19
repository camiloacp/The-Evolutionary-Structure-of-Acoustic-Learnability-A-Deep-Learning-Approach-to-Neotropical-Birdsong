import pandas as pd

def sinonimias(df: pd.DataFrame, avonet1: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces specific species names in the 'Species1' column of the 'avonet1' DataFrame 
    and the 'Especie' column of the 'df' DataFrame with their corresponding synonyms.
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the 'Especie' column.
    - avonet1 (pd.DataFrame): The DataFrame containing the 'Species1' column.
    Returns:
    - pd.DataFrame: The modified 'df' DataFrame with replaced species names in the 'Especie' column.
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

    df['Especie'] = df['Especie'].str.replace("Corapipo_altera", "Corapipo_leucorrhoa")
    df['Especie'] = df['Especie'].str.replace("Xenops_rutilans", "Xenops_rutilus")
    
    return df, avonet1
    