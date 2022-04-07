from googletrans import Translator
from tqdm import tqdm 
import pandas as pd 
import nltk
from nltk.corpus import stopwords



import gensim.downloader as gensim_api


class NLP():
    def __init__(self):
        self.gensim_glve = gensim_api.load('glove-wiki-gigaword-300')
        try:
            self.stwd = stopwords.words('english')
        except:
            nltk.download('stopwords')
            self.stwd = stopwords.words('english')
        self.translator = Translator()
        self.ko_stopwords = list(pd.read_excel("data/ko_stopwords.xlsx").T.values[0])
        self.en_stopwords = list(pd.read_excel("data/en_stopwords.xlsx").T.values[0])

    def propose_similar_words(self, words:list, topn=5, korean=True, model="glove"):
        if model == "glove":
            model = self.gensim_glve
        else:
            raise ValueError(f"{model} is not supported")
        
        # translate specific language to other
        if korean:
            print("[INFO] korean language is translated by googletrans")
            translated_words = self.translator.translate(words, dest="en")
            en_words = [w.text.lower() for w in translated_words]
        else:
            en_words = words
        
        similar_en_words = [model.most_similar(positive=[word],topn=topn) for word in en_words ]
        if korean:
            similar_kr_words = [[words[k]] + [j.text for j in self.translator.translate(i, dest="ko")]  for k,i in tqdm(enumerate(similar_en_words))]
        else:
            similar_kr_words = None

        return similar_kr_words, similar_en_words

