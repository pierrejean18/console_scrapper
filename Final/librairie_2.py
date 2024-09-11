"""
La librairie permet, à partir de la page d'accueil des résultats de la recherche des "consoles", 
d'obtenir une liste de liens en fonction
du résultat de filtrage d'une catégorie spécifique, choisie par défaut.

>>> from librairie_2 import ScrapperLienProduit
>>> sc=ScrapperLienProduit()
>>> l=sc.liens_produits_categorie()
>>> l
[<librairie_2.PagesProduitsCategories at 0x16d90e44970>,
 <librairie_2.PagesProduitsCategories at 0x16d90e747c0>]
>>> l[0].cat_produit
'Console jeux vidéo'
>>> l[0].liens_produits
['https://bons-plans.easycash.fr/console-jeux-video/sony-ps1-gris-001120964',
 'https://bons-plans.easycash.fr/console-jeux-video/nintendo-wii-blanc-wii-sport-001171039',
 ...
  "'https://bons-plans.easycash.fr/console-jeux-video/microsoft-xbox-one-1-to-noir-001907034']
"""
from typing import List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from time import sleep
import librairie_1 as l1


class PagesProduitsCategories:
    cat_produit: str
    liens_produits: List[str]

    def __init__(self, cat_produit: str, liens_produits: List[str]):
        self.cat_produit = cat_produit
        self.liens_produits = liens_produits


sc=l1.EasyCashScraper()
l=sc.differente_categorie()


class ScrapperLienProduit:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()

    def _page_suivante(self):
        """Clic sur la page suivante si c'est possible."""
        next_button = self.driver.find_element(
            By.CSS_SELECTOR, "ul.list-pagination.listing-products > li.next > a")
        self.driver.execute_script(
            "arguments[0].scrollIntoView();", next_button)
        sleep(3)
        WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "ul.list-pagination.listing-products > li.next > a")))
        next_button.click()
        sleep(3)

    def _lien_produit_categorie(
            self, categorie: l1.PageCategorie) -> PagesProduitsCategories:
        """Renvoie une liste de liens vers les détails de tous les produits d'une
        catégorie spécifique."""
        pages_produit_categorie = PagesProduitsCategories(
            cat_produit=categorie.cat, liens_produits=[]
        )
        while True:
            try:
                produits_page = self.driver.find_elements(
                    By.CSS_SELECTOR, "li.clearfix.block-link")
                for produit in produits_page:
                    produit = produit.find_element(
                        By.CSS_SELECTOR, "a.link-buy").get_attribute("href")
                    pages_produit_categorie.liens_produits.append(produit)
                self._page_suivante()
            except NoSuchElementException:
                break
        return pages_produit_categorie


    def liens_produits_categorie(
        self, Liste_Categorie: List[l1.PageCategorie]=l
    ) -> List[PagesProduitsCategories]:
        """Renvoie une liste d'objet qui contiens une listes des liens
        vers les détails de chaque produits d'une catégorie spécifique."""
        Liste_liens_produits = []
        for categorie in Liste_Categorie:
            self.driver.get(categorie.lien)
            l1.accept_cookies(self.driver)
            Liste_liens_produits.append(
                self._lien_produit_categorie(categorie))
        self.driver.quit()
        return Liste_liens_produits
