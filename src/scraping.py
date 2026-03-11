import random
import time

from bs4 import BeautifulSoup
import pandas as pd
import requests

from config import DATA_DIR


def scrape_trustpilot(url: str, output_file: str):
    data = []

    i = 1
    visited = set()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 '
        'Safari/537.36'
    }
    while True:
        print(f'Scraping page {i}')
        url = f'{url}{i}'
        response = requests.get(url, headers=headers)

        if response.url in visited:
            break

        if response.status_code == 200:
            print('Success')
        else:
            print(f'Failed to retrieve the page, status code: {response.status_code}')
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        reviews = soup.find_all(
            'div', class_='styles_cardWrapper__g8amG styles_show__Z8n7u'
        )

        page_data = []
        for r in reviews:
            date = r.find(
                'span',
                class_='CDS_Typography_appearance-inherit__68c681 '
                'CDS_Typography_prettyStyle__68c681 CDS_Typography_body-'
                's__68c681 CDS_Typography_disableResponsiveSizing__68c681 '
                'CDS_Badge_badgeText__083901',
            )
            if date:
                date = date.text
            text = r.find(
                'p',
                class_='CDS_Typography_appearance-default__68c681 '
                'CDS_Typography_prettyStyle__68c681 CDS_Typography_body-l__68c681',
            )
            if text:
                text = text.text
            score = r.find('div', class_='styles_reviewHeader__DzoAZ')
            if score:
                score = score['data-service-review-rating']

            page_data.append((date, score, text))

        if len(page_data) == 0:
            break
        data.extend(page_data)

        i += 1
        time.sleep(random.uniform(1, 3))

    df = pd.DataFrame(data, columns=['Date', 'Score', 'Review'])
    df.to_csv(DATA_DIR / output_file)


if __name__ == '__main__':
    scrape_trustpilot(
        'https://www.trustpilot.com/review/www.samsung.com/uk?languages=en&page=2',
        'reviews_uk.csv',
    )
