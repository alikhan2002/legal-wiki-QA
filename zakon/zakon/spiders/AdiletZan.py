import scrapy
from bs4 import BeautifulSoup, NavigableString, Tag
import re

from urllib.parse import urlencode


class AdiletzanSpider(scrapy.Spider):
    name = "AdiletZan"
    start_urls = ["https://adilet.zan.kz/kaz/index/docs"]
    def parse(self, response):
        hrefs = response.css('div.serp a').getall()
        links = []
        for html_snippet in hrefs:

            soup = BeautifulSoup(html_snippet, 'html.parser')

            href_value = soup.find('a')['href']
            links.append(href_value)

        for link in links:
            yield response.follow(
                f'https://adilet.zan.kz{link}',
                callback=self.parse_article
            )
        next_page = response.css('a.nextpostslink::attr(href)').get()
        if next_page is not None:
            next_page_url = 'https://adilet.zan.kz' + next_page
            yield response.follow(next_page_url, callback=self.parse)




    def parse_article(self, response):
        html_content = response.css('[class="container_alpha slogan"]').get()
        soup = BeautifulSoup(html_content, 'html.parser')
        self.title = soup.get_text(separator='\n', strip=True)

        html_content = response.css("div.gs_12").get()
        if 'Күшін жойған' in self.title:
            return
        soup = BeautifulSoup(html_content, 'html.parser')

        articles = soup.find_all('article')
        contents = []
        for article in articles:  # Iterate over each article
            for element in article.children:  # Use .descendants to iterate through all children, recursively
                if element.name == 'table':
                    for row in element.find_all('tr'):
                        row_data = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                        contents.append('\t|\t'.join(row_data) + '\n')
                    contents.append('\n')

                elif element.name == 'a':
                    link_text = element.get_text(strip=True)
                    if link_text:
                        contents.append(link_text + ' ')
                elif element.name == 'span':
                    span_text = element.get_text(strip=True)
                    if span_text:
                        contents.append(span_text + ' ')
                    # print(span_text)
                #
                else:
                    text_content = element.get_text(strip=True, separator=' ')
                    contents.append(text_content + '\n')
        cleaned_name = re.sub(r'[\\/*?:"<>|\r\n]', " ", self.title)
        data = {
            cleaned_name: ''.join(contents)
        }
        yield data

    # def to_txt(self):

    #     temp = self.d['title'] + '\n\n' + self.d['text']
    #     dir = 'C:/Users/aliha/Desktop/NLP/parser/zakon/check/'
    #     cleaned_name = re.sub(r'[\\/*?:"<>|\r\n]', " ", self.title)[:100] + ".txt"
    #     self.names.append(cleaned_name)
    #     path = dir + cleaned_name
    #     with open(f'{path}', 'w', encoding='utf-8') as f:
    #         f.write(temp)


#