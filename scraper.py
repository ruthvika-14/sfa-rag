import requests
from bs4 import BeautifulSoup

URLS = [
    "https://safefoodalliance.com/",
    "https://safefoodalliance.com/about/",
    "https://safefoodalliance.com/safe-food-certifications/",
    "https://safefoodalliance.com/about/food-safety-certifications-and-accreditations/",
    "https://safefoodalliance.com/audits/internal-audit-checklist-for-haccp-brcgs-and-sqf/",
    "https://safefoodalliance.com/industry-updates/dfa-of-california-forms-a-new-business-segment/",
]

def scrape_all():
    docs = []
    for url in URLS:
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            docs.append({"url": url, "text": text})
            print(f"Scraped: {url}")
        except Exception as e:
            print(f"Failed: {url} -- {e}")
    return docs

if __name__ == "__main__":
    docs = scrape_all()
    for doc in docs:
        print(f"\n--- {doc['url']} ---")
        print(doc['text'][:300])