# import chainlit as cl
# from dotenv import load_dotenv
# import os
# from assistant import Assistant

# global assistant

# def pretty_print(par_titles, pars, ref_list, query):
#     answer_str = f'Ответ по запросу: {query}\n'

#     for t,p in zip(par_titles, pars):
#         answer_str += t + '\n' + '. '.join(p) + '\n\n'

#     answer_str += 'СПИСОК ИСПОЛЬЗОВАННОЙ ЛИТЕРАТУРЫ:\n'

#     for r in ref_list:
#         answer_str += r + '\n\n'

#     return answer_str

# @cl.on_chat_start
# async def on_chat_start():
#     load_dotenv()
#     api_key = os.getenv('API_KEY')

#     assistant = Assistant("faiss_index/", api_key)

#     runnable = assistant

#     cl.user_session.set("runnable", runnable)

# @cl.on_message
# async def main(message : str):

#     assistant = cl.user_session.get("runnable")

#     par_titles, pars, ref_list = await assistant.get_answer(message.content)

#     response = pretty_print(par_titles, pars, ref_list, message.content)

#     await cl.Message(content = response).send()










import chainlit as cl
from dotenv import load_dotenv
import os
from assistant import Assistant
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
global assistant
from lingua import Language, LanguageDetectorBuilder
import requests 
import time

def pretty_print(par_titles, pars, ref_list, query, kz):
    if kz:
        answer_str = f'Сұраққа жауап: {query}\n'
    else:
        answer_str = f'Ответ по запросу: {query}\n'

    for t,p in zip(par_titles, pars):
        answer_str += t + '\n' + '. '.join(p) + '\n\n'
    if kz:
        answer_str += 'ҚОЛДАНЫЛҒАН ӘДЕБИЕТТЕР:\n'
    else:
        answer_str += 'СПИСОК ИСПОЛЬЗОВАННОЙ ЛИТЕРАТУРЫ:\n'

    for r in ref_list:
        answer_str += r + '\n\n'

    return answer_str

def detect(text):
    languages = [Language.RUSSIAN, Language.KAZAKH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    confidence_values = detector.compute_language_confidence_values(text)
    # confidence_values
    for confidence in confidence_values:
        if confidence.language.name == "KAZAKH" and confidence.value >= 0.3:
            return 'kz'
    return 'ru'
 
def translate(text, reverse=True): 
    text_in_url = '+'.join(text.split()) 
    if reverse == False: 
        url = f"https://t.song.work/api?text={text_in_url}&from=kk&to=ru" 
    else: 
        url = f"https://t.song.work/api?text={text_in_url}&from=ru&to=kk" 
    print(text, url)
    
    response =  requests.get(url) 
    time.sleep(0.01)
    
    response_json = response.json() 
    
    result = response_json['result'] 
    
    return result


@cl.on_chat_start
async def on_chat_start():
    load_dotenv()
    api_key = os.getenv('API_KEY')
    assistant = Assistant("final_db/", api_key)
    runnable = assistant
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def main(message : str):

    assistant = cl.user_session.get("runnable")
    kz = (detect(message.content) == 'kz')
    text = message.content
    if kz:
        text =  translate(message.content, False)
    par_titles, pars, ref_list = await assistant.get_answer(text)

    if kz:
        # print(par_titles)
        # print()
        # print(pars)
        # print()
        # print(ref_list)
        # print()
        par_titles =  [ translate(txt) for txt in par_titles]
        pars =  [[ translate(txt) for txt in par] for par in pars ]
        ref_list =   [ translate(txt) for txt in ref_list]

    response = pretty_print(par_titles, pars, ref_list, message.content, kz)


    await cl.Message(content = response).send()