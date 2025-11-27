import re
from bs4 import BeautifulSoup
from selenium import webdriver

url = 'https://www.nike.com/in/help/'

driver = webdriver.Chrome()
driver.get(url)
html = driver.page_source
driver.quit()


soup = BeautifulSoup(html, "html.parser")

for tag in soup.select("nav, header, footer, #nav-menu, .nav"):
    tag.decompose()

anchors = soup.find_all("a")   # all <a> tags

print(len(anchors))

for a in anchors:
    print(a.get("href"), a.get_text(strip=True))