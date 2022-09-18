import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask

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

    # Test code
    print(rating_data.head(2))
    print(music_data.head(2))

    return "Rating data size {}".format(str(rating_data.size))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
