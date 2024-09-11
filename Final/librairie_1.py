"""Cette Librairie nous permet d'obtenir en fonction de la catégorie filtrée, le lien
de la page des résultats associé au filtrage de cette catégorie ainsi que son nom.
>>> sc=EasyCashScraper()
>>> l=sc.different_categorie()
>>> l
[<librairie_1.PageCategorie at 0x16d90b823a0>,
 <librairie_1.PageCategorie at 0x16d90b825b0>]
>>> l[0].cat
'Console jeux vidéo'
>>> l[0].lien
'https://bons-plans.easycash.fr/catalog/search?filterType=searchResults&q=consoles&facets%5BsubCategory%5D%5B%5D=Console+jeux+vid%C3%A9o&minPrice=&maxPrice='

"""

from typing import List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from time import sleep


class PageCategorie:
    cat: str
    lien: str


def accept_cookies(driver):
    """"accepte les cookies si le bouton est présent."""
    try:
        cookies = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        cookies.click()
    except TimeoutException:
        pass


class EasyCashScraper:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()

    def recherche(self, terme_rechercher: str = "consoles") -> str:
        """recherche le terme par défault "consoles" sur la barre de recherche et renvoie 
         le liens de la page des résultats. """
        self.driver.get("https://bons-plans.easycash.fr")
        accept_cookies(self.driver)
        search_box = self.driver.find_element(By.ID, "search-field")
        search_box.send_keys(terme_rechercher)
        search_box.submit()
        return self.driver.current_url

    def click_categorie(self):
        """Clic sur le filtre catégorie et fait défiler le choix des catégorie possible."""
        try:
            filter_button = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "#side-core > div.inner > ul:nth-child(4) > li:nth-child(2)",
                    )
                )
            )
            filter_button.click()
        except (TimeoutException, NoSuchElementException):
            print("Erreur : Le filtre catégorie n'est pas présent sur cette page.")
            raise

    def page_categorie(self, i: int):
        """"Nous permet d'obtenir le nom de la catégorie que l'on a séléctionner
        ainsi que le lien de la page des résultats associer au filtrage de cette catégorie."""
        css_selector = f"#side-content-cat > ul > li:nth-child({i})"
        filter_option = self.driver.find_element(By.CSS_SELECTOR, css_selector)
        categorie = PageCategorie()
        categorie.cat = filter_option.get_attribute("data-label")
        self.driver.execute_script("arguments[0].scrollIntoView();", filter_option)
        WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, css_selector))
        )
        sleep(1)
        filter_option.click()
        WebDriverWait(self.driver, 20).until(EC.staleness_of(filter_option))
        categorie.lien = self.driver.current_url
        return categorie

    def differente_categorie(self, j: int = 2, k: int = 4) -> List[PageCategorie]:
        """Nous permet d'obtenir une liste d'objet de type PageCategorie,
        qui contiens pour chaque catégorie le nom de la catégorie que l'on a filtrer ainsi 
        que le lien des résultats associer."""
        liste_categories = []
        lien_h = (
            self.recherche()
        )  # ouvre le lien car back() ne ramene pas à la bonne page
        try:
            for i in range(j, k):
                accept_cookies(self.driver)
                self.click_categorie()
                categorie = self.page_categorie(i)
                liste_categories.append(categorie)
                self.driver.get(lien_h)
        finally:
            self.driver.quit()
        return liste_categories
