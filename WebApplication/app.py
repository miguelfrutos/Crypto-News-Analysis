# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 02:17:51 2022

@author: jpthoma
"""

from flask import Flask,render_template

app = Flask(__name__,template_folder="templates")

@app.route("/")
#@app.route("/home")

def home():
    return render_template("index.html")


if __name__=="__main__":
    app.run(debug=True)