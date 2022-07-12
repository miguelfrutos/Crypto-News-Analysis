# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 02:17:51 2022

@author: jpthoma
"""

from flask import Flask,render_template
import pandas as pd
from gdeltdoc import GdeltDoc, Filters
from datetime import date, timedelta, datetime

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

    daybefore = currenday - timedelta(days=1)
    start_date = daybefore.strftime("%Y-%m-%d")
    end_date = currenday.strftime("%Y-%m-%d")
    start_date,end_date
    
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

    tokens_df = tokenizer.batch_encode_plus(
        df.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

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
    lista.append(probs)
    
    # SENTIMENT
    #load weights of best model
    path_sentiment = '/Users/miguelfrutossoriano/Desktop/git/crypto/Crypto_Curated_Database/04_Models/saved_weights_sentiment.pt'
    model.load_state_dict(torch.load(path_sentiment))
    with torch.no_grad():
        logits = model(val_seq, val_mask)
        probs = F.softmax(logits, dim=1) 
    lista.append(probs)
    
    # STRENGHT
    #load weights of best model
    path_strenght = '/Users/miguelfrutossoriano/Desktop/git/crypto/Crypto_Curated_Database/04_Models/saved_weights_strenght.pt'
    model.load_state_dict(torch.load(path_strenght))
    with torch.no_grad():
        logits = model(val_seq, val_mask)
        probs = F.softmax(logits, dim=1) 
    lista.append(probs)



    #rendering the output in the index.html file
    return render_template('index.html', prediction_text=lista_pred)
    


if __name__=="__main__":
    app.run(debug=True)