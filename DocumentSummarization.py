# Libraries for pdf conversion
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import re
from io import StringIO

# Libraries for feature extraction and topic modeling
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import mglearn

# Visualization
import pyLDAvis
import pyLDAvis.sklearn
from os import path
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS

# Other libraries
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

class DocumentSummarization():

   """
   Class for the summarizing SEC filings
   """
   def __init__(self, document, text_output_file):
      self.document = document
      self.text_output_file = text_output_file
      self.text = ""

   def convert_pdf_to_txt(self):
      # Params for pdf parsing
      rsrcmgr = PDFResourceManager()
      retstr = StringIO()
      laparams = LAParams()
      device = TextConverter(rsrcmgr, retstr, laparams=laparams)
      fp = open(self.document, 'rb')
      interpreter = PDFPageInterpreter(rsrcmgr, device)
      password = ""
      maxpages = 0
      caching = True
      pagenos=set()

      for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
         interpreter.process_page(page)

      # Grab the text
      text = retstr.getvalue()
      # self.text = retstr.getvalue()

      fp.close()
      device.close()
      retstr.close()
      self.text = text
      return text

   def create_text_file(self):
      # Write the data from the conversion from convert_pdf_to_txt to given filename
      f=open(self.text_output_file,'w')
      f.write(self.text)
      f.close()

   def clean_text_file(self):
      with open(self.text_output_file) as f:
         clean_cont = f.read().splitlines()
         doc=[i.replace('\xe2\x80\x9c','') for i in clean_cont ]
         doc=[i.replace('\xe2\x80\x9d','') for i in doc ]
         doc=[i.replace('\xe2\x80\x99s','') for i in doc ]

         docs = [x for x in doc if x != ' ']
         docss = [x for x in docs if x != '']
         financedoc=[re.sub("[^a-zA-Z]+", " ", s) for s in docss]
         return financedoc
         
   def create_pyldavis_model(self, clean_text):
      vect=CountVectorizer(ngram_range=(1,1),stop_words='english')
      fin=vect.fit_transform(clean_text)
      pd.DataFrame(fin.toarray(),columns=vect.get_feature_names()).head(1)
      lda=LatentDirichletAllocation(n_components=5)
      lda.fit_transform(fin)
      sorting=np.argsort(lda.components_)[:,::-1]
      array=np.full((1, sorting.shape[1]), 1)
      array = np.concatenate((array,sorting), axis=0)

      zit=pyLDAvis.sklearn.prepare(lda,fin,vect)
      return zit
      # different ways to display: https://pyldavis.readthedocs.io/en/latest/modules/API.html
      # for now we will just return zit and display in notebook manually, html option for flask app
      # pyLDAvis.show(zit)
      # JN: pyLDAvis.display(doc.create_pyldavis_model(clean_text))

   def create_word_cloud(self):
      text = open(self.text_output_file).read()
      stopwords = set(STOPWORDS)
      wc = WordCloud(background_color="black", max_words=2000, stopwords=stopwords)
      wc.generate(text)
      plt.figure(figsize=(16,13))
      plt.imshow(wc, interpolation='bilinear')
      plt.axis("off")
      plt.figure()
      plt.axis("off")
      plt.show()