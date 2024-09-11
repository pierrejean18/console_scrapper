import pytest
from Final.librairie_1 import EasyCashScraper
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from Final.librairie_1 import accept_cookies, PageCategorie

scraper = EasyCashScraper()


#@pytest.fixture(scope="module")
#def driver():
#    driver = webdriver.Chrome()
 #   driver.maximize_window()
  #  yield driver
   # driver.quit()


def test_recherche()
    resultat = scraper.recherche()
    assert resultat == "https://bons-plans.easycash.fr/catalog/search?q=consoles"


def test_click_categorie():
    scraper.driver.get("https://bons-plans.easycash.fr/catalog/search?q=consoles")
    accept_cookies(scraper.driver)
    scraper.click_categorie()


@pytest.mark.xfail
def test_click_categorie2():
    scraper.driver.get(
        "https://bons-plans.easycash.fr/console-jeux-video/sony-ps4-1-to-noir-001906201"
    )
    accept_cookies(scraper.driver)
    scraper.click_categorie()


def test_click_categorie3():
    scraper.driver.get(
        "https://bons-plans.easycash.fr/console-jeux-video/sony-ps4-1-to-noir-001906201"
    )
    with pytest.raises(TimeoutException):
        accept_cookies(scraper.driver)
        scraper.click_categorie()


def test_accept_cookies_present(driver):
    driver.get("https://www.easycash.fr/")
    accept_cookies(driver)


def test_page_categorie_cat():
    scraper.driver.get(
        "https://bons-plans.easycash.fr/catalog/search?filterType=searchResults&q=consoles"
    )
    accept_cookies(scraper.driver)
    scraper.page_categorie
    scraper.click_categorie()
    obj = scraper.page_categorie(2)
    assert obj.cat == "Console jeux vidéo"


def test_page_categorie_lien():
    scraper.driver.get(
        "https://bons-plans.easycash.fr/catalog/search?filterType=searchResults&q=consoles"
    )
    accept_cookies(scraper.driver)
    scraper.page_categorie
    scraper.click_categorie()
    obj = scraper.page_categorie(2)
    assert (
        obj.lien
        == "https://bons-plans.easycash.fr/catalog/search?filterType=searchResults&q=consoles&facets%5BsubCategory%5D%5B%5D=Console+jeux+vid%C3%A9o&minPrice=&maxPrice="
    )


def test_type_objet_page_categorie():
    scraper.driver.get(
        "https://bons-plans.easycash.fr/catalog/search?filterType=searchResults&q=consoles"
    )
    accept_cookies(scraper.driver)
    scraper.page_categorie
    scraper.click_categorie()
    obj = scraper.page_categorie(2)
    assert isinstance(obj, PageCategorie)


def test_differente_categorie_len_liste():
    liste_test = scraper.differente_categorie()
    assert len(liste_test) == 2


def test_diffrente_categorie_1_cat():
    liste_test = scraper.differente_categorie()
    assert liste_test[1].cat == "Console portable"


def test_differente_categorie_1_lien():
    liste_test = scraper.differente_categorie()
    assert (
        liste_test[1].lien
        == "https://bons-plans.easycash.fr/catalog/search?filterType=searchResults&q=consoles&facets%5BsubCategory%5D%5B%5D=Console+portable&minPrice=&maxPrice="
    )


def test_obj_differente_categorie():
    liste_test = scraper.differente_categorie()
    assert isinstance(liste_test[0], PageCategorie)
