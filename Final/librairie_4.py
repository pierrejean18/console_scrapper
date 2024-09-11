""""
Librairie qui nous permet d'obtenir une liste de dictionnaire contenant 
les caractérisitques de tous les produits.
>>> from librairie_4 import CodeSourceProduitsScraper
>>> sc=CodeSourceProduitsScraper()
>>> l=sc.list_all_produits()
>>> l[1]
{'État': 'Bon état',
 'Prix': '299,99 \n\n',
 'Couleur': 'Noir',
 'Nb de manette(s) incluse(s)': '2',
 'Nom commercial': 'Switch',
 'Edition Limitée': 'OUI',
 'Date de sortie': '2023',
 'Console en pack': 'NON',
 'Plateforme': 'Switch',
 'Jeu fourni': 'Aucun',
 'Capacité': '64 Go',
 'WI-FI intégré': 'OUI',
 'Connexion Ethernet/LAN': 'OUI',
 'Bluetooth': 'OUI',
 'Entrée jack microphone': 'OUI',
 'Port HDMI': 'OUI',
 'Accessoires supp.': 'NON',
 'Jeux vidéos inclus': 'NON'}

"""

from typing import List, Dict
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from bs4.element import Tag
import librairie_3 as l3


class CodeSourceProduits:
    cat_produits_csource = str
    code_source_produits = List[str]

    def __init__(self, cat_produits_csource: str, code_source_produits: List[str]): #a modifier
        self.cat_produits_csource = cat_produits_csource
        self.code_source_produits = code_source_produits


l = l3.source_produits()


class CodeSourceProduitsScraper:
    @staticmethod
    def accept_cookies2(driver:webdriver.chrome.webdriver.WebDriver):
        """Fonction qui permet d'accepter les cookies sur une page uniquement si le bouton 
        pour accepter les cookies est présent"""
        try:
            cookies = WebDriverWait(driver, 1).until(
                EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
            )
            cookies.click()
        except TimeoutException:
            pass

    @staticmethod
    def extract_info_etat_prix(balise_a: Tag, produit_i: Dict):
        """Extrait le prix et l'état du produit."""
        grade_label_span = balise_a.find("span", {"class": "grade-label"})
        grade_price_span = balise_a.find("span", {"class": "grade-price"})
        if grade_label_span and grade_price_span:
            produit_i["État"] = grade_label_span.text
            produit_i["Prix"] = grade_price_span.text

    @staticmethod
    def characteristics(code_source: Tag, produit_i: Dict):
        """Extrait toutes les caractérisitques technique du produit."""
        table = code_source.find("table")
        for row in table.find_all("tr"):
            columns = row.find_all("td")
            key, value = (
                columns[0].get_text(strip=True),
                columns[1].get_text(strip=True),
            )
            produit_i[key] = value

    @staticmethod
    def nom_produit(code_source: Tag, produit_i: Dict):
        """Extrait le nom du produit."""
        nom_produit = code_source.find("h1", {"class": "block-product--title"})
        produit_i["nom"] = nom_produit.text

    @classmethod
    def list_sous_produits(
        cls, code_source: Tag, code_source_categorie: CodeSourceProduits
    )-> List:
        """Extrait les imformation pour les produits potentiel sur une page.
        (sur une page il peut y avoir un produit ou plusieurs.)"""
        div_tab_links = code_source.find("div", {"class": "tab_links"})
        Liste_sous_produits = []
        for balise_a in div_tab_links.find_all("a"):
            produit_i = {}
            cls.extract_info_etat_prix(balise_a, produit_i)
            cls.characteristics(code_source, produit_i)
            cls.nom_produit(code_source, produit_i)
            produit_i[
                "consoles_portable_unq"
            ] = code_source_categorie.cat_produits_csource
            Liste_sous_produits.append(produit_i)
        return Liste_sous_produits

    @classmethod
    def list_all_produits(cls, codest: List[CodeSourceProduits] = l) -> List[Dict]:
        """Extrait les imformations de tous les produits."""
        liste_produits = []
        for code_source_categorie in codest:
            for code_source in code_source_categorie.code_source_produits:
                liste_produits.extend(
                    cls.list_sous_produits(code_source, code_source_categorie)
                )
        return liste_produits
