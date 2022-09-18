import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, jsonify, request

app = Flask(__name__)

# Get File From CloudFront
def getFileFromCloudFront(cloudFrontUrl, filePath):
  file = requests.get(cloudFrontUrl + filePath, allow_redirects=True)
  url_content = file.content
  csv_file = open(filePath,'wb')
  csv_file.write(url_content)
  csv_file.close()

# For Sorting
def f1(x):
	return x[1]

# test code
@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return "Hello {}!".format(name)

@app.route("/recommendCF", methods=['POST'])
def recommend():
    # 1) Get data from CloudFront

    # Get CloudFront url from env
    cloudFrontUrl = os.environ.get("CLOUD_FRONT_URL")

    # Get rating data
    ratingsPath = "ratings.csv"
    getFileFromCloudFront(cloudFrontUrl, ratingsPath)
    rating_data = pd.read_csv(ratingsPath)

    # Get music data
    musicsPath = "musics.csv"
    getFileFromCloudFront(cloudFrontUrl, musicsPath)
    music_data = pd.read_csv(musicsPath)

    # 2) Make CF Model

    # Drop timestamp data
    rating_data.drop('timestamp', axis = 1, inplace=True)

    # Merge user data and music data
    user_music_rating = pd.merge(rating_data, music_data, on = 'musicId')

    # Make pivot table
    music_user_rating = user_music_rating.pivot_table('rating', index = 'title', columns='userId')
    user_music_rating = user_music_rating.pivot_table('rating', index = 'userId', columns='title')

    # Fill NA to 0
    music_user_rating.fillna(0, inplace = True)

    # Get cosine similarity
    item_based_collabor = cosine_similarity(music_user_rating)

    # Get dataframe
    item_based_collabor = pd.DataFrame(data = item_based_collabor, index = music_user_rating.index, columns = music_user_rating.index)

    # 3) Inference

    # Get users data from request
    params = request.get_json()
    arr =params['musicArr']
    print(arr)

    rec = dict() # recommended idx
    ret = []

    for title in arr:
        title = int(title)
        try:
            recommends = item_based_collabor[title].sort_values(ascending=False)[:6]
            idxs = recommends.index
            print(idxs)
            for idx in idxs:
                if(idx == title or recommends[idx] == 1): # pass input and unnecessary data
                    continue
                if(str(idx) in arr): # pass all input
                    continue 
                if(idx in rec): # already recommend
                    rec[idx] += 1
                else:           # new recommend
                    rec[idx] = 1
            
            # sorting by value
            sortedArr = sorted(rec.items(),key=f1)

            for item in sortedArr:
                ret.append(item[0])
                if(len(ret) == 20):
                    break                
        except Exception as e:
            print(e)

    print(ret)

    return jsonify(ret)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
