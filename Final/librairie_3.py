"""La librairie nous permet d'obtenir une liste de code source pour chaque produit,
associé à une catégorie spécifique.

>>> from librairie_3 import source_produits
>>> l3=source_produits()
>>> l3
[<librairie_3.CodeSourceProduits at 0x22516124250>,
 <librairie_3.CodeSourceProduits at 0x22516145760>]
>>> l3[0].code_source_produits[0]
<html class="no-js easycash-v3 new-css" lang="fr" xmlns:fb="https://www.facebook.com/2008/fbml" xmlns:og="http://ogp.me/ns#"><head>
<link href="https://auixaysgjr.cloudimg.io" rel="preconnect"/>
<link href="https://auixaysgjr.cloudimg.io" rel="dns-prefetch"/>
<link href="https://cdn.cookielaw.org" rel="preconnect"/>
<link href="https://cdn.cookielaw.org" rel="dns-prefetch"/>
...
<script async="" src="https://cdn.speedcurve.com/js/lux.js?id=4345843475"></script></body></html>
>>> l3[0].cat_produits_csource
'Console jeux vidéo'

"""

from typing import List
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import librairie_2 as l2


class CodeSourceProduits:
    cat_produits_csource = str
    code_source_produits = List[str]

    def __init__(self, cat_produits_csource: str, code_source_produits: List[str]):
        self.cat_produits_csource = cat_produits_csource
        self.code_source_produits = code_source_produits


def accept_cookies2(driver):
    """"accepte les cookies si ils sont présent et attend 1 seconde maximum pour trouver le bouton"""
    try:
        cookies = WebDriverWait(driver, 1).until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        cookies.click()
    except TimeoutException:
        pass


sc = l2.ScrapperLienProduit()
l = sc.liens_produits_categorie()


def source_produits(
    Liste_liens_produits: List[l2.PagesProduitsCategories] = l,
) -> List[CodeSourceProduits]:
    """Renvoie une liste de tous les codes sources de tous les produits pour chaque catégorie spécifique."""
    Liste_Code_Source = []
    for liens_produits in Liste_liens_produits:
        codesource = CodeSourceProduits(
            cat_produits_csource=liens_produits.cat_produit, code_source_produits=[]
        )
        driver = webdriver.Chrome()
        driver.maximize_window()
        for lien in liens_produits.liens_produits:
            driver.get(lien)
            accept_cookies2(driver)
            codesource.code_source_produits.append(bs(driver.page_source, "lxml"))
        Liste_Code_Source.append(codesource)
    return Liste_Code_Source
