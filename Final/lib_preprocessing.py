import json
import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer

chemin_fichier_json = 'final\liste_final.json'
data_brut = []

with open(chemin_fichier_json, 'r') as f:
    data_brut = json.load(f)
    
nouveau_cle_nom = 'Marque'
for dictionnaire in data_brut:
    dictionnaire[nouveau_cle_nom] = dictionnaire.pop('nom')
    
nouveau_cle_nom = 'Consoles_portable_unq'
for dictionnaire in data_brut:
    dictionnaire[nouveau_cle_nom] = dictionnaire.pop('consoles_portable_unq')

df = pd.DataFrame(data_brut)

def price(df):
    """Modifie la présentation de `Prix`"""
    colonne_Prix = 'Prix'
    df[colonne_Prix] = df[colonne_Prix].str.replace(r' \n\n', '')
    return df

def color(df):
    """Modifie la présentation de `Couleur`"""
    colonne_Couleur = 'Couleur'
    df[colonne_Couleur] = df[colonne_Couleur].str.replace(r', ', '-')
    df['Couleur'] = df['Couleur'].apply(lambda x: 'Autres' if x != 'Noir' and x != 'Blanc' and x != 'Gris' else x)
    return df

def nb_manette(df):
    """Modifie la présentation de `Nb de manette(s) incluse(s)`"""
    def update_manettes(row):
        if row['Consoles_portable_unq'] == 'OUI' and row['Nb de manette(s) incluse(s)'] == 'NA':
            return 'cons_sans_manette'
        else:
            return row['Nb de manette(s) incluse(s)']
    
    df['Nb de manette(s) incluse(s)'] = df.apply(update_manettes, axis=1)
    return df

def release_date(df):
    """Modifie la présentation de `Date de sortie`"""
    df['Date de sortie'] = df['Date de sortie'].replace(['2020', '2021', '2022', '2023'], '2020-2023')
    df['Date de sortie'] = df['Date de sortie'].replace(['2017', '2018', '2019'], '2017-2019')
    df['Date de sortie'] = df['Date de sortie'].replace(['2016', '2015'], '2015-2016')
    df['Date de sortie'] = df['Date de sortie'].replace(['2012', '2013', '2014'], '2012-2014')
    df['Date de sortie'] = df['Date de sortie'].replace(['2010', '2011'], '2010-2011')
    df['Date de sortie'] = df['Date de sortie'].replace(['2005', '2006', '2007', '2008', '2009'], '2005-2009')
    return df

def supply_game(df):
    """Modifie la présentation de `Jeu fourni`"""
    df["Jeu fourni"] = df["Jeu fourni"].replace(['NON', 'Aucun'], 'NON')
    df["Jeu fourni"] = df["Jeu fourni"].apply(lambda x: 'OUI' if x != 'NON' else x)
    return df

def capacity(df):
    """Modifie la présentation de `Capacité`"""
    df['Capacité'] = df['Capacité'].str.replace(r' Go', '')
    df['Capacité'] = df['Capacité'].str.replace(r'Aucune', '0')
    df['Capacité'] = df['Capacité'].str.replace(r' To', '000')
    df['Capacité'] = df['Capacité'].str.replace(r'N.C.', 'NA')
    df['Capacité'] = df['Capacité'].apply(lambda x: '1' if re.search(r'1\D', x) else x)
    df['Capacité'] = df['Capacité'].apply(lambda x: '2' if re.search(r'2\D', x) else x)
    return df

def additional_accessories(df):
    """Modifie la présentation de `Accessoires supp.`"""
    df["Accessoires supp."] = df["Accessoires supp."].replace(['Aucun'], 'NON')
    df["Accessoires supp."] = df["Accessoires supp."].replace(['N.C.'], 'NA')
    df["Accessoires supp."] = df["Accessoires supp."].apply(lambda x: 'OUI' if x != 'NON' and x != 'NA' else x)
    return df

def brand(df):
    """Modifie la présentation de `Marque`"""
    df["Marque"] = df["Marque"].apply(lambda x: 'Autres' if x != 'Sony' and x != 'Nintendo' and x != 'Microsoft' else x)
    return df

def dimensions(df):
    """Modifie la présentation de `Dimensions (LxPxH)`"""
    df['Dimensions (LxPxH)'] = df['Dimensions (LxPxH)'].str.replace(r' mm', '')
    df['Dimensions (LxPxH)'] = df['Dimensions (LxPxH)'].str.replace(r'\xa0', '')
    df['Dimensions (LxPxH)'] = df['Dimensions (LxPxH)'].apply(lambda x: x if x == 'NA' else re.sub(r'[^0-9x]', ' ', x))
    return df

def height(df):
    """Modifie la présentation de `Hauteur`"""
    df['Hauteur'] = df['Hauteur'].str.replace(r' mm', '')
    return df

def weight(df):
    """Modifie la présentation de `Poids`"""
    df['Poids'] = df['Poids'].str.replace(' Kg', '')
    df['Poids'] = df['Poids'].str.replace(' kg', '')
    return df

def width(df):
    """Modifie la présentation de `Largeur`"""
    df["Largeur"] = df["Largeur"].str.replace(r' mm', '')
    return df

def cleaning(df):
    """Nettoie la présentation de notre dataframe"""
    df = price(df)
    df = color(df)
    df = nb_manette(df)
    df = release_date(df)
    df = supply_game(df)
    df = capacity(df)
    df = additional_accessories(df)
    df = brand(df)
    df = dimensions(df)
    df = height(df)
    df = weight(df)
    df = width(df)
    return df

df = cleaning(df)

df_clean = df[['Marque', 'Consoles_portable_unq',
                'État', 'Prix', 'Couleur', 'Nb de manette(s) incluse(s)',
                'Edition Limitée', 'Date de sortie', 'Console en pack',
                'Jeu fourni', 'Capacité', 'WI-FI intégré',
                'Bluetooth', 'Accessoires supp.', 'Jeux vidéos inclus']]

def encode_category(df_clean):
    """Recode les variables catégorielles en category et les variables continues en numérique"""
    df_clean['Marque'] = df_clean['Marque'].astype('category')
    df_clean['Consoles_portable_unq'] = df_clean['Consoles_portable_unq'].astype('category')
    df_clean['État'] = df_clean['État'].astype('category')
    df_clean['Couleur'] = df_clean['Couleur'].astype('category')
    df_clean['Nb de manette(s) incluse(s)'] = df_clean['Nb de manette(s) incluse(s)'].astype('category')
    df_clean['Edition Limitée'] = df_clean['Edition Limitée'].astype('category')
    df_clean['Date de sortie'] = df_clean['Date de sortie'].astype('category')
    df_clean['Console en pack'] = df_clean['Console en pack'].astype('category')
    df_clean['Jeu fourni'] = df_clean['Jeu fourni'].astype('category')
    df_clean['WI-FI intégré'] = df_clean['WI-FI intégré'].astype('category')
    df_clean['Bluetooth'] = df_clean['Bluetooth'].astype('category')
    df_clean['Accessoires supp.'] = df_clean['Accessoires supp.'].astype('category')
    df_clean['Jeux vidéos inclus'] = df_clean['Jeux vidéos inclus'].astype('category')
    
    df_clean['Prix'] = pd.to_numeric(df_clean['Prix'].str.replace(',', '.'), errors='coerce')
    df_clean['Capacité'] = pd.to_numeric(df_clean['Capacité'], errors='coerce')
    return df_clean

df_clean = encode_category(df_clean)

def numeric_impute(df_clean):
    """Impute les variables numériques"""
    columns_to_impute_numeric = ['Capacité']
    df_clean[columns_to_impute_numeric] = df_clean[columns_to_impute_numeric].apply(lambda col: col.replace('NA', np.nan))
    imputer = SimpleImputer(strategy='mean')
    df_clean[columns_to_impute_numeric] = imputer.fit_transform(df_clean[columns_to_impute_numeric])
    return df_clean

df_clean = numeric_impute(df_clean)

def categorical_impute(df_clean):
    """Impute les variables catégorielles"""
    columns_to_impute_categorical = ['Nb de manette(s) incluse(s)', 'Console en pack', 'WI-FI intégré', 'Bluetooth', 'Accessoires supp.']
    df_clean[columns_to_impute_categorical] = df_clean[columns_to_impute_categorical].apply(lambda col: col.replace('NA', np.nan))
    
    columns_to_impute_categorical = ['Nb de manette(s) incluse(s)', 'Console en pack', 'WI-FI intégré', 'Bluetooth', 'Accessoires supp.']
    return df_clean

df_clean = categorical_impute(df_clean)

def encode_category_again(df_clean):
    """Recode à nouveau les variables"""
    df_clean['Nb de manette(s) incluse(s)'] = df_clean['Nb de manette(s) incluse(s)'].astype('category')
    df_clean['Console en pack'] = df_clean['Console en pack'].astype('category')
    df_clean['WI-FI intégré'] = df_clean['WI-FI intégré'].astype('category')
    df_clean['Bluetooth'] = df_clean['Bluetooth'].astype('category')
    df_clean['Accessoires supp.'] = df_clean['Accessoires supp.'].astype('category')
    return df_clean

df_clean = encode_category_again(df_clean)

df_clean.to_csv('df_clean.csv', index=False)
