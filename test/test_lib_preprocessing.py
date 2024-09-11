import pytest
import json
import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
import sys
sys.path.append(r"C:\Users\karl\Downloads\GHAFFOUR_JEAN_SONDEJI_PROJECT")
from final.lib_preprocessing import encode_category, numeric_impute, categorical_impute


def test_encode_category():
    """Vérifier le type des variables encodées"""
    data = {
        'Marque': ['Nintendo', 'Sony', 'Microsoft'],
        'État': ['Bon état', 'Parfait état', 'Bon état'],
        'Couleur': ['Blanc', 'Noir', 'Gris'],
        'Prix': ['100,00', '200,00', '150,00'],
        'Capacité': ['500GB', '1TB', '250GB']
    }
    df = pd.DataFrame(data)

    df_encoded = encode_category(df)

    assert df_encoded['Marque'].dtype == 'category'
    assert df_encoded['État'].dtype == 'category'
    assert df_encoded['Couleur'].dtype == 'category'

    assert pd.api.types.is_numeric_dtype(df_encoded['Prix'])
    assert pd.api.types.is_numeric_dtype(df_encoded['Capacité'])


""" test_encode_category() """

def test_numeric_impute():
    """Vérifier l'imputation des variables numériques"""
    data = {
        'Marque': ['Nintendo', 'Sony', 'Microsoft'],
        'Capacité': ['500GB', '1TB', 'NA']
    }
    df = pd.DataFrame(data)
    df_imputed = numeric_impute(df)
    assert not df_imputed['Capacité'].isna().any()


""" test_numeric_impute() """

def test_categorical_impute():
    """Vérifier l'imputation des variables catégorielles"""
    data = {
        'Marque': ['Nintendo', 'Sony', 'Microsoft'],
        'Nb de manette(s) incluse(s)': ['2', 'NA', '1'],
        'Console en pack': ['OUI', 'NA', 'NON'],
        'WI-FI intégré': ['OUI', 'NA', 'OUI'],
        'Bluetooth': ['NON', 'NA', 'OUI'],
        'Accessoires supp.': ['NA', 'OUI', 'NA']
    }
    df = pd.DataFrame(data)
    df_imputed = categorical_impute(df)

    assert df_imputed[['Nb de manette(s) incluse(s)', 'Console en pack', 'WI-FI intégré', 'Bluetooth', 'Accessoires supp.']].isna().any().any()

""" test_categorical_impute() """