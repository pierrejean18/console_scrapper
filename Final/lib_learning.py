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

df = pd.read_csv('final\df_clean.csv')

def recode_ml(df):
    marques_recode = {'Nintendo': 1, 'Sony': 2, 'Microsoft': 3, 'Autres': 4}
    df['Marque'] = df['Marque'].replace(marques_recode)
    cns_portable_recode = {'NON': 0, 'OUI': 1}
    df['Consoles_portable_unq'] = df['Consoles_portable_unq'].replace(cns_portable_recode)
    etat_recode = {'Bon état': 0, 'Parfait état': 1}
    df['État'] = df['État'].replace(etat_recode)
    couleurs_recode = {'Blanc': 1, 'Noir': 2, 'Gris': 3, 'Autres': 4}
    df['Couleur'] = df['Couleur'].replace(couleurs_recode)
    nb_manettes_recode = {'0': 0, '1': 1, '2': 2, 'cons_sans_manette': 3}
    df['Nb de manette(s) incluse(s)'] = df['Nb de manette(s) incluse(s)'].replace(nb_manettes_recode)
    edition_limitée_recode = {'NON': 0, 'OUI': 1}
    df['Edition Limitée'] = df['Edition Limitée'].replace(edition_limitée_recode)
    release_date_recode = {'Avant 2005': 0, '2005-2009': 1, '2010-2011': 2,
                            '2012-2014': 3, '2015-2016': 4, '2017-2019': 5, '2020-2023': 6}
    df['Date de sortie'] = df['Date de sortie'].replace(release_date_recode)
    console_pack_recode = {'NON': 0, 'OUI': 1}
    df['Console en pack'] = df['Console en pack'].replace(console_pack_recode)
    jeu_fourni_recode = {'NON': 0, 'OUI': 1}
    df['Jeu fourni'] = df['Jeu fourni'].replace(jeu_fourni_recode)
    wifi_recode = {'NON': 0, 'OUI': 1}
    df['WI-FI intégré'] = df['WI-FI intégré'].replace(wifi_recode)
    bluetooth_recode = {'NON': 0, 'OUI': 1}
    df['Bluetooth'] = df['Bluetooth'].replace(bluetooth_recode)
    accessoires_supp_recode = {'NON': 0, 'OUI': 1}
    df['Accessoires supp.'] = df['Accessoires supp.'].replace(accessoires_supp_recode)
    jeux_inclus_recode = {'NON': 0, 'OUI': 1}
    df['Jeux vidéos inclus'] = df['Jeux vidéos inclus'].replace(jeux_inclus_recode)
    return df

df = recode_ml(df)

def one_hot_tranform(df):
    # Définissez les colonnes catégorielles que vous souhaitez encoder
    categorical_columns = ['Marque', 'Consoles_portable_unq', 'État', 'Couleur',
                            'Nb de manette(s) incluse(s)', 'Edition Limitée', 
                            'Date de sortie', 'Console en pack', 'Jeu fourni', 
                            'WI-FI intégré', 'Bluetooth', 'Accessoires supp.']

    # Créez un transformateur pour appliquer One-Hot Encoding aux colonnes catégorielles
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Créez un transformateur pour appliquer les transformations sur l'ensemble des colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns)
        ],
        remainder='passthrough'  # Cela permet de conserver les colonnes non catégorielles inchangées
    )

    # Créez un pipeline avec les transformations
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Appliquez les transformations sur le dataframe
    df_encoded = pd.DataFrame(pipeline.fit_transform(df))

    # Remplacez les noms de colonnes par les noms originaux
    column_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_columns)
    df_encoded.columns = list(column_names) + list(df.columns[len(categorical_columns):])

    return df_encoded

df_encoded = recode_ml(df)
df_encoded["Prix"] = df["Prix"]

y = df_encoded.iloc[:,-1].values
X = df_encoded.iloc[:,:-1].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=85)

y = df["Prix"].values
X = df[['Marque', 'Consoles_portable_unq',
    'État', 'Couleur', 'Nb de manette(s) incluse(s)',
    'Edition Limitée', 'Date de sortie', 'Console en pack',
    'Jeu fourni', 'WI-FI intégré', 'Bluetooth',
    'Accessoires supp.', 'Jeux vidéos inclus']].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=85)

onehot_encoder = OneHotEncoder()
onehot_encoder.fit(X_tr)
X_train = onehot_encoder.transform(X_tr)
X_test = onehot_encoder.transform(X_te)

def training(df):
    """Fonction qui permet d'entraîner le modèles et renvoies en sortie deux tableaux
    l'un faisant la synthèse des modèles et l'autre présentant les résultats de la prédiction.
    La fonction met environ 5 minutes pour entraîner les modèles"""

    # Regression linéaire

    lr = LinearRegression()
    lr_final = lr.fit(X_tr, y_tr)

    # Lasso

    ls = Lasso()
    ls_final = ls.fit(X_tr, y_tr)

    # Ridge

    ri = Ridge()
    ri_final = ri.fit(X_tr, y_tr)

    # Elastic Net

    en = ElasticNet()
    en_gs = GridSearchCV(
        en,
        {
            "alpha": [2**p for p in range(-6, 6)],
            "l1_ratio": (0.01, 0.25, 0.5, 0.75, 1),
        },
        n_jobs=-1
    )
    en_gs_final = en_gs.fit(X_tr, y_tr)

    # K Nearest Neighbors

    knr = KNeighborsRegressor()
    knr_gs = GridSearchCV(
        knr,
        {
            "n_neighbors": (2, 4, 8, 16, 32),
            "weights": ("uniform", "distance"),
        },
        n_jobs=-1
    )
    knr_gs_final = knr_gs.fit(X_tr, y_tr)

    # Gaussian process Regression

    gpr = GaussianProcessRegressor()
    gpr_final = gpr.fit(X_tr, y_tr)

    # Random Forest

    rfr = RandomForestRegressor()
    rfr_gs = GridSearchCV(
        rfr,
        {
            "n_estimators": (16, 32, 64, 128, 256),
            "max_depth": (1, 10, 50, 100, None),
            "min_samples_leaf": (1, 2, 5, 10),
            "max_features": ["auto", "sqrt", "log2"],
        },
        n_jobs=-1
    )
    rfr_gs_final = rfr_gs.fit(X_tr, y_tr)

    # Support Vector Regression

    pl = Pipeline([("mise_echelle", MinMaxScaler()), ("support_vecteurs", SVR())])
    pl_gs = GridSearchCV(
        pl,
        {
            "support_vecteurs__C": (0.1, 1.0, 10),
            "support_vecteurs__epsilon": (0.1, 1.0, 10),
        },
        n_jobs=-1
    )
    svr_gs_final = pl_gs.fit(X_tr, y_tr)

    # Multi Layer Perceptron

    pln = Pipeline([("mise_echelle", MinMaxScaler()), ("neurones", MLPRegressor())])

    pln_gs = GridSearchCV(
        pln,
        {
            "neurones__alpha": 10.0 ** -np.arange(1, 7),
            "neurones__hidden_layer_sizes": ((25,), (50,), (100,), (20, 20)),
        },
        n_jobs=-1
    )
    mlp_gs_final = pln_gs.fit(X_tr, y_tr)

    # XGBoost

    gs_boost = GridSearchCV(
        XGBRegressor(),
        {
            "nthread": [4],
            "objective": ["reg:linear"],
            "learning_rate": [0.03, 0.05, 0.07],
            "max_depth": [5, 6, 7],
            "min_child_weight": [4],
            "silent": [1],
            "subsample": [0.7],
            "colsample_bytree": [0.7],
            "n_estimators": [500],
        },
        n_jobs=-1
    )

    xgb_gs_final = gs_boost.fit(X_tr, y_tr)

    # Gradient boosting regressor

    gb_gs = GridSearchCV(
        GradientBoostingRegressor(),
        {
            "n_estimators": [50, 100, 500],
            "learning_rate": [0.01, 0.1, 1.0],
            "subsample": [0.5, 0.7, 1.0],
            "max_depth": [3, 7, 9],
        },
        n_jobs=-1
    )
    gb_gs_final = gb_gs.fit(X_tr, y_tr)

    # Adaboost

    adb_gs = GridSearchCV(
        AdaBoostRegressor(),
        {"learning_rate": [0.01, 0.1, 1.0], "n_estimators": [50, 100, 150, 200]},
        n_jobs=-1
    )

    adb_gs_final = adb_gs.fit(X_tr, y_tr)

    # MLP network
    mlp = GridSearchCV(
    MLPRegressor(),
    {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 100)],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [100, 150, 200],
    },
    n_jobs=-1
)
    
    mlp_final = mlp.fit(X_tr, y_tr)

    models = [
        lr_final,
        ls_final,
        ri_final,
        en_gs_final,
        knr_gs_final,
        gpr_final,
        rfr_gs_final,
        svr_gs_final,
        mlp_gs_final,
        xgb_gs_final,
        gb_gs_final,
        adb_gs_final,
        mlp_final
    ]
    return models

list_models = training(df)

list_best_models = []
for modele in list_models:
    if isinstance(modele, GridSearchCV):
        # Si le modèle est une instance de GridSearchCV
        meilleur_modele = modele.best_estimator_
    else:
        meilleur_modele = modele  # Si le modèle n'est pas une instance de GridSearchCV

    # Assurez-vous que les données d'entraînement et de test sont en format dense
    if issparse(X_train):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train

    if issparse(X_test):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test

    # Entraîner le meilleur modèle sur les données d'entraînement
    modele_train = meilleur_modele.fit(X_train_dense, y_tr)
    list_best_models.append(modele_train)
    # Prédire sur les données de test
    predictions = meilleur_modele.predict(X_test_dense)
    # Calculer l'erreur (ou toute autre métrique)
    erreur = mean_squared_error(y_te, predictions)

    print(f"Erreur pour le modèle {type(meilleur_modele).__name__}: {erreur}")
    
for model in list_best_models:
    if issparse(X_train):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train

    if issparse(X_test):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
    if "GridSearchCV" in str(model):
        model_name = str(model.estimator)
        best_params = model.best_params_
        best_score = model.score(X_train_dense, y_tr)
        if "Pipeline" in str(model):
            model_name = str(model.estimator[1])
    else:
        model_name = str(model)
        best_score = model.score(X_train_dense, y_tr)
        best_params = "/"
    print(model_name, best_score, best_params)

def predicts(list_models):
    for model in list_models:
        # Obtenez les prédictions pour le modèle actuel
        if issparse(X_test):
            X_test_dense = X_test.toarray()
        else:
            X_test_dense = X_test
        y_pred = model.predict(X_test_dense)  # Assurez-vous d'avoir X_te correctement défini

        # Convertissez les prédictions à la forme souhaitée
        true = np.array(y_te)
        pred = np.array(y_pred)
        pred = np.around(pred, decimals=1)

    dataframe_data = []

    # Ajoutez des lignes à la liste (exemple)
    df['id'] = range(1, len(df) + 1)
    for id in df['id']:
        x = df.index[df['id'] == id].astype(int)
        for i in x:
            if 0 <= i < len(true):
                dataframe_data.append({
                    "Réalité": f"{true[i]} €",
                    "Prédiction": f"{pred[i]:.1f} €",
                    "Ecart": f"{((pred[i] - true[i]) / true[i]) * 100:.2f} %",
                })

    # Créez un DataFrame à partir des données
    df_resultat = pd.DataFrame(dataframe_data)

    # Enregistrez le DataFrame en tant qu'image (par exemple, en CSV ici)
    df_resultat.to_csv("resultat.csv", index=False)

    return df_resultat

predicts(list_models)