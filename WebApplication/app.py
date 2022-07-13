# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 02:17:51 2022

@author: jpthoma
"""

import pandas as pd
from gdeltdoc import GdeltDoc, Filters
from datetime import date, timedelta, datetime
from transformers import  BertTokenizerFast
import torch
import torch.nn as nn
#!pip install gdeltdoc
#!pip install transformers
#pip install selenium

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)


app = Flask(__name__,template_folder="templates")

@app.route("/")
#@app.route("/home")
def home():
    return render_template("index.html")

#Set a post method to yield predictions on page
@app.route('/', methods = ['POST'])
def predict():
    # INGESTION FROM GDELT
    ## Select the period to ingest
    currenday = date.today()- timedelta(days=30)
    daybefore = currenday - timedelta(days=7)
    start_date = daybefore.strftime("%Y-%m-%d")
    end_date = currenday.strftime("%Y-%m-%d")
    
    # Filtering the period
    f = Filters(keyword='bitcoin', # HERE THE FILTER
    start_date = str(start_date),
    end_date = str(end_date))
    
    #GDELT object
    gd = GdeltDoc()

    # Search for articles matching the filters
    articles = gd.article_search(f)
    df = articles[(articles["language"] == 'English')]
    df = df['title']

    # ADAPTING INPUTS

    # Load the BERT tokenizer
    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('ProsusAI/finbert')
    tokenizer = BertTokenizerFast.from_pretrained('ProsusAI/finbert')
    # Tokenize
    tokens_df = tokenizer.batch_encode_plus(
        df.tolist(),
        max_length = 40,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    #Seq and Mask tensors
    val_seq = torch.tensor(tokens_df['input_ids'])
    val_mask = torch.tensor(tokens_df['attention_mask'])
    
    # Prepare the Output
    lista_pred = []

    # RELEVANCE
    #load weights of best model
    path_relevance = '/Users/miguelfrutossoriano/Desktop/git/crypto/Crypto_Curated_Database/04_Models/saved_weights_relevance.pt'
    model.load_state_dict(torch.load(path_relevance))
    # get predictions for val data
    with torch.no_grad():
        logits = model(val_seq, val_mask)
        probs = F.softmax(logits, dim=1) 
    lista_pred.append(probs)
    
    # SENTIMENT
    #load weights of best model
    path_sentiment = '/Users/miguelfrutossoriano/Desktop/git/crypto/Crypto_Curated_Database/04_Models/saved_weights_sentiment.pt'
    model.load_state_dict(torch.load(path_sentiment))
    with torch.no_grad():
        logits = model(val_seq, val_mask)
        probs = F.softmax(logits, dim=1) 
    lista_pred.append(probs)
    
    # STRENGHT
    #load weights of best model
    path_strenght = '/Users/miguelfrutossoriano/Desktop/git/crypto/Crypto_Curated_Database/04_Models/saved_weights_strenght.pt'
    model.load_state_dict(torch.load(path_strenght))
    with torch.no_grad():
        logits = model(val_seq, val_mask)
        probs = F.softmax(logits, dim=1) 
    lista_pred.append(probs)

    #rendering the output in the index.html file
    return render_template('index.html', prediction_text=lista_pred)
    


if __name__=="__main__":
    app.run(debug=True)