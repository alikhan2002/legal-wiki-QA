from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from itertools import chain
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
#from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import sent_tokenize
import fasttext as ft
from collections import defaultdict
#from greed import greed_sum_query

from tqdm.notebook import tqdm
import os

import pickle
def save_obj(obj, name):
    pickle.dump(obj,open(name + '.pkl', 'wb'), protocol=4)
    
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
sw = stopwords.words('russian')

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings("ignore")
import backoff
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import openai
import re
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from razdel import tokenize, sentenize
import string
import fasttext as ft

class Assistant:
    
    def __init__(self, path, api_key):
        self.model = ft.load_model("cc.ru.300.bin")
        self.embedding = OpenAIEmbeddings(api_key=api_key, model='text-embedding-3-large')
        self.db = FAISS.load_local("final_db/", embeddings = self.embedding, allow_dangerous_deserialization=True)
        self.sw = stopwords.words('russian')

    async def greed_sum_query(self, text, query_vec, query, num_sent=10, min_df=1, max_df=20):
        # Let's take 10% of the most meaningful sentences
        #num_sent = 10 # int(len(text)*0.05) 
        #print('Number of 5% sentences', num_sent)
        #fit a TFIDF vectorizer
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=self.sw)
        X1 = vectorizer.fit_transform(text).toarray()

        # query specific TFIDF
        voc = set(word_tokenize(query.lower()))

    # Initialize the second vectorizer with the unique vocabulary
        vectorizer2 = TfidfVectorizer(max_df=max_df, vocabulary=list(voc))
        # voc = word_tokenize(query.lower())
        # vectorizer2 = TfidfVectorizer(max_df=max_df,vocabulary=voc)
        X2 = vectorizer2.fit_transform(text).toarray()

        X = np.concatenate((X1, X2), axis=1)

        text_vecs = np.array([self.model.get_sentence_vector(s.replace('\n','')) for s in text])

        
        similarity = 1 - pairwise_distances(query_vec.reshape(1, -1), text_vecs, metric='cosine')
        X = X * similarity[0][:, None]

        idx = []
        while sum(sum(X)) != 0:
            ind = np.argmax(X.sum(axis=1))
            idx.append(ind)
            #stop if we have more than 20% of the sentences from the text
            if len(idx) > int(len(text)*0.2):#num_sent:
                break

            #update the matrix deleting the columns corresponding to the words found in previous step
            cols = X[ind]
            col_idx = [i for i in range(len(cols)) if cols[i] > 0]
            X = np.delete(X, col_idx, 1)

        #make a condition to extract a number of sentences or all salient sentences
        if num_sent != 0:
            idx = idx[:num_sent]
        idx.sort()
        summary = [text[i] for i in idx]
        summary_vecs = [text_vecs[i] for i in idx]
        return summary, summary_vecs

    def clean_text(self, str):
        str = re.sub('[\xa0\n]', ' ', str)
        str = str.replace('№', 'номер')
        return str

    def clean_query(self, str):
        clean_query = ""
        for i in self.clean_text(str.lower()).split(" "):
            if i not in self.sw and i not in string.punctuation:
                clean_query += i + " " 
                
        clean_query = clean_query.strip()

        return clean_query

    # def get_section_refs(self, sources, texts):
    #     statyas = []
    #     for name, text in zip(sources, texts):
    #         search_term = text.encode('utf-8')  # Note: binary string for mmap
    #         with open(fr'.\Data\{name}.txt','r',encoding='utf-8') as file:
    #             content = file.read()
    #             index = content.find(text)
    #             pattern = r'Статья \d+'
    #             match = re.search(pattern, content[index:index+12])
    #             index += 12
    #             for s in range(index, 10, -1):
    #                 temp = content[s-11:s] 
    #                 match = re.search(pattern, temp)
    #                 if match:
    #                     statya = match.group()
    #                     statyas.append(statya)
    #                     break
    #     return statyas

    def get_refs(self, final_summary_texts, t_summarized_each):
        refs = set()
        for s in final_summary_texts:
            for i, d in enumerate(t_summarized_each) :
                if s in self.clean_text(d):
                    refs.add(i)
        return refs

    def paragraphs_split(self, final_sum):
        sum_sent_vecs = [np.array(self.model.get_sentence_vector(s.replace('\n',''))) for s in final_sum] 
        sims = []
        for i in range(1, len(sum_sent_vecs)):
            sims.append(1 - pairwise_distances(sum_sent_vecs[i].reshape(1, -1), sum_sent_vecs[i-1].reshape(1, -1), metric='cosine')[0][0])
        treshld = 0.5
        par_begin = np.array(sims) <= treshld
        par_begin_idx = np.array(range(len(sims)))[par_begin]
        return par_begin_idx
    
    async def get_answer(self, q):
        #GREED SUMMARY
        query = self.clean_query(q)
        query_vec = self.model.get_sentence_vector(query) 
        
        docs = self.db.similarity_search(query=query, k = 10)
        t = [ i.page_content for i in docs ]
        sources = [ i.metadata['source'].split('\\')[-1].split('.')[0] for i in docs ]
        #Summarizing each section
        t_summarized_each, sentences_t_summarized_together, sentences_t_summarized_together_vecs = ([] for i in range(3))
        for section in t:
            section_summmary, section_summmary_vec = await self.greed_sum_query([i.text for i in list(sentenize(self.clean_text(section))) ], query_vec, query, num_sent=10)
            t_summarized_each.append(" ".join(section_summmary))
            sentences_t_summarized_together += section_summmary
            sentences_t_summarized_together_vecs += section_summmary_vec

        text_vecs = np.array(sentences_t_summarized_together_vecs)
        
        similarity = 1 - pairwise_distances(query_vec.reshape(1, -1), text_vecs, metric='cosine')
#         print(similarity)
        indexes = [i[1] for i in sorted([(j, i) for i, j in enumerate(similarity[0])], reverse=True) if i[0] >= 0.4]
        final_summary_texts = [sentences_t_summarized_together[i] for i in indexes]
        # statyas = self.get_section_refs(sources, t)
        
        # 4. Get references for each sentence in the resulting summary
        refs = self.get_refs(final_summary_texts, t_summarized_each)
                    
        final_sum = [s + '[%s]'%(r+1) for s, r in zip(final_summary_texts, refs)]
        ref_list = ['%s. '%(r+1)+sources[r] for r in refs]
        # 5. Separate the text into paragraphs
        ## Find paragraph boundaries
    
        ### consecutive sentence pair similarity
        par_begin_idx = self.paragraphs_split(final_sum)

        pars = self.partition(final_sum, list(par_begin_idx))
        par_tits = [self.get_title(p, query) for p in pars]
        return par_tits, pars, ref_list
    
    
    ## Make titles for paragraphs
    def get_title(self, par, query):
        
        vectorizer = TfidfVectorizer(ngram_range=(3, 3),stop_words=sw)
        X = vectorizer.fit_transform(par).toarray().sum(axis=0)

        titles = vectorizer.get_feature_names_out()

        #query_vec =  np.array(embedding.embed_query(query)) #
        query_vec = self.model.get_sentence_vector(query) 
        #text_vecs =  np.array([embedding.embed_query(s) for s in titles]) #
        text_vecs = np.array([self.model.get_sentence_vector(s.replace('\n','')) for s in titles])

        similarity = 1 - pairwise_distances(query_vec.reshape(1, -1), text_vecs, metric='cosine')

        return titles[np.argmax(similarity)].upper() 
    
    def partition(self, alist, indices):
        return [alist[i:j] for i, j in zip([0]+indices, indices+[None])]
