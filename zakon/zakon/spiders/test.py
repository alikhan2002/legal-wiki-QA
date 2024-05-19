# # from bs4 import BeautifulSoup
# #
# # # It's good practice to use a path that doesn't hard-code the user directory.
# # # If this script is meant to be used on different machines, consider making the file path relative or configurable.
# # file_path = r'C:\Users\aliha\Desktop\NLP\parser\zakon\ddd.txt'
# #
# # with open(file_path, 'r', encoding='utf-8') as f:
# #     text = f.read()
# #
# # soup = BeautifulSoup(text, 'html.parser')
# #
# # # Your initial approach tries to find all <article> tags but doesn't use them afterwards.
# # # Assuming you want to process tables within these articles:
# # articles = soup.find_all('article')
# #
# # # It's better to specify the file's encoding when writing, especially if you expect non-ASCII characters.
# # with open('dat@.txt', 'w', encoding='utf-8') as f:  # Corrected the file name to match the output message
# #     for article in articles:  # Iterate over each article
# #         for element in article.children:  # Use .descendants to iterate through all children, recursively
# #             if element.name == 'table':
# #                 for row in element.find_all('tr'):
# #                     row_data = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
# #                     f.write('\t|\t'.join(row_data) + '\n')
# #                 f.write('\n')
# #             elif element.name == 'a':
# #                 link_text = element.get_text(separator=' ')
# #                 if link_text:
# #                     f.write(link_text)
# #             elif element.name == 'span':
# #                 span_text = element.get_text(separator=' ')
# #                 if span_text:
# #                     f.write(span_text + ' ')
# #             else:
# #                 text_content = element.get_text(strip=True)
# #                 f.write(text_content + '\n')
# #
# # print("Data has been saved to data.txt")
# import os
# with open('C:/Users/aliha/Desktop/NLP/parser/zakon/names/names4.txt', 'r', encoding='utf-8') as f:
#     names = f.read().split('\n')
# dir_list = os.listdir('C:/Users/aliha/Desktop/NLP/parser/zakon/current/')
# dir_list = [''.join(filename.split())[:50] for filename in dir_list if filename.endswith(".txt")]
# mx = max(len(s) for s in dir_list)
# print(dir_list)
# # for i in range(len(dir_list)):
# #     if dir_list[i][:len('Конституция')] == 'Конституция':
# #         print(i)
# print(dir_list[808])
# for i in names:
#     # print(i.strip())
#     if i[:50] not in dir_list:
#         print(i[:50])
#         break
#
# # from bs4 import BeautifulSoup
# # import re
# # html_content = '<div class="post_holder">\n\t\t\t\t\t\n\t\t\t\t\t\t\t\t\n\t\t\t\n\t\t\t<h4 class="post_header">\n\t\t\t\t<span class="post_number">1.</span>\n\t\t\t\t<a href="/rus/docs/K950001000_">Конституция Республики Казахстан</a>\n\t\t\t</h4>\n\t\t\n\t\t\t\t\t\t\t\t\n\t\t\t<span class="status status_upd">Обновленный</span>\n\t\t\t\n\t\t\t\n\t\t\n\t\t\t\t\t\t\t\t<p>Конституция принята на республиканском референдуме 30 августа 1995 года.</p>\n\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\n\t\t\t\t</div>'
# # soup = BeautifulSoup(html_content, 'html.parser')
# #
# # # Extracting text from the provided HTML content
# # extracted_text = soup.get_text().strip()
# # cleaned_text = re.sub(r'\n+', ' ', extracted_text)
# # cleaned_text = re.sub(r'^\d+\.', '', cleaned_text).strip()
# # cleaned_name = re.sub(r'[\\/*?:"<>|\r\n]', " ", cleaned_text)[:100]
# #
# # print(cleaned_name)
# # # Конституция Республики Казахстан Обновленный Конституция принята на республиканском референдуме 30
# # # Конституция Республики Казахстан Обновленный Конституция принята на республиканском референдуме 30 а