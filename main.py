from os import read
from flask import Flask,jsonify,request
import csv

all_articles = []

with open('articles.csv')as f:
    reader = csv.reader(f)
    data = list(reader)
    all_articles = data[1:]

liked_articles = []
disLiked_articls = []

#creating constructor
app = Flask(__name__)

@app.route('/get-article')
def gettheArticles():
    return jsonify({
        'data':all_articles[0],
        'status':'success'
    })
@app.rout('/like-articles',methods = ['POST'])
def likedArticle():
    article = all_articles[0]
    liked_articles.append(article)
    all_articles.pop(0)
    return jsonify({
        'status':'success'
    }),201

@app.route('/dislike-articles',methods = ['POST'])
def dislikedArticle():
    article = all_articles[0]
    disLiked_articls.append(article)
    all_articles.pop(0)    
    return jsonify({
        'status':'success'
    }),201


if __name__ == '__main__':
    app.run()
