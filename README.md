# Projet de Web Scraping et Prédiction de Prix des Consoles

Ce projet développe un système de **web scraping** afin de prédire les prix des consoles de jeux vidéo. Nous avons conçu quatre librairies pour extraire les données d'un site web, qui sont ensuite stockées dans un fichier `liste_final.json`. 

## Prétraitement des Données

Les données extraites sont nettoyées et unifiées à l'aide de la bibliothèque `Pandas`. Ce processus inclut la concaténation de plusieurs fichiers JSON pour former un tableau de données structuré, prêt pour l'analyse.

## Modélisation

Ensuite, plusieurs algorithmes de **machine learning** sont entraînés pour prédire les prix des consoles. Parmi les modèles utilisés, nous avons :
- **Gradient Boosting**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Multi-Layer Perceptron (MLP)**

Ces modèles sont optimisés par validation croisée pour trouver les paramètres les plus performants. Le meilleur modèle est sélectionné pour réaliser des prédictions sur les données de test.

## Résultats

Les prédictions finales sont stockées dans un fichier CSV, `resultat.csv`. Le processus de scraping peut être relancé, ce qui déclenchera un réentraînement des modèles pour s'assurer que les données les plus récentes sont prises en compte.

L'objectif global de ce projet est de fournir un pipeline complet, allant de l'extraction des données à la prédiction, en passant par la modélisation et l'analyse.

