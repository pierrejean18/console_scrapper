import pytest
import pandas as pd
import numpy as np
from rich import print
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from scipy.sparse import issparse
import sys
sys.path.append(r"C:\Users\karl\Downloads\GHAFFOUR_JEAN_SONDEJI_PROJECT")
from final.lib_learning import recode_ml, one_hot_tranform

@pytest.fixture
def sample_df():
    data = {
        'Marque': ['Nintendo', 'Sony', 'Microsoft', 'Autres'],
        'Consoles_portable_unq': ['OUI', 'NON', 'OUI', 'OUI'],
        'État': ['Parfait état', 'Bon état', 'Bon état', 'Parfait état'],
        # Ajoutez d'autres colonnes selon votre besoin
    }
    return pd.DataFrame(data)

def test_recode_ml(sample_df):
    """Vérifier que les variables ont été correctement recoder"""
    result_df = recode_ml(sample_df)

    assert result_df['Marque'].equals(pd.Series([1, 2, 3, 4]))
    assert result_df['Consoles_portable_unq'].equals(pd.Series([1, 0, 1, 1]))
    assert result_df['État'].equals(pd.Series([1, 0, 0, 1]))
    

def test_one_hot_transform():
    """Vérifier que les transformations ont été bien effectuées"""
    data = {
        'Marque': ['Nintendo', 'Sony', 'Microsoft'],
        'État': ['Bon état', 'Parfait état', 'Bon état'],
        'Couleur': ['Blanc', 'Noir', 'Gris'],
        'Prix': [100, 200, 150]
    }
    df = pd.DataFrame(data)

    df_encoded = one_hot_tranform(df)

    expected_columns = ['Marque_Microsoft', 'Marque_Nintendo', 'Marque_Sony',
                        'État_Bon état', 'État_Parfait état',
                        'Couleur_Blanc', 'Couleur_Gris', 'Couleur_Noir', 'Prix']
    assert np.array_equal(df_encoded.columns, expected_columns)

    assert df_encoded['Marque_Nintendo'].iloc[0] == 1
    assert df_encoded['Marque_Sony'].iloc[1] == 1
    assert df_encoded['État_Parfait état'].iloc[1] == 1


""" test_one_hot_transform() """
