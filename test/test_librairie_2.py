import pytest
from Final.librairie_2 import ScrapperLienProduit,PagesProduitsCategories
from selenium import webdriver


sc=ScrapperLienProduit()
l=sc.liens_produits_categorie()

@pytest.fixture(scope="module")
def sc():
    sc = ScrapperLienProduit()
    yield sc
    sc.driver.quit()

@pytest.fixture(scope="module")
def liens_produits(sc):
    return sc.liens_produits_categorie()

def test_verification(liens_produits):
    assert len(liens_produits) == 2
    assert liens_produits[0].cat_produit == 'Console jeux vidéo'
    assert len(liens_produits[0].liens_produits)==322
    assert liens_produits[0].liens_produits[0]=='https://bons-plans.easycash.fr/console-jeux-video/sony-ps1-gris-001120964'


@pytest.mark.xfail
def test_click_page_suivante():
    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get("https://bons-plans.easycash.fr/catalog/search?filterType=searchResults&q=consoles&facets%5BsubCategory%5D%5B0%5D=Console+jeux+vid%C3%A9o&offset=330")
    sc._page_suivante(driver)

def test_click_page_suivante():
    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get("https://bons-plans.easycash.fr/catalog/search?q=consoles")
    sc._page_suivante(driver)
