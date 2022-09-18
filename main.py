import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, jsonify

app = Flask(__name__)

def getFileFromCloudFront(cloudFrontUrl, filePath):
  file = requests.get(cloudFrontUrl + filePath, allow_redirects=True)
  url_content = file.content
  csv_file = open(filePath,'wb')
  csv_file.write(url_content)
  csv_file.close()

# test code
@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return "Hello {}!".format(name)

@app.route("/recommendCF")
def recommend():
    # Get data from CloudFront

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

    # Inference
    title = 54825
    recommends = item_based_collabor[title].sort_values(ascending=False)[:6]

    print(recommends)

    return jsonify(recommends.to_json)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
