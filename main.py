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

# test code
@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return "Hello {}!".format(name)

@app.route("/recommendCF")
def recommend():
    
    # test env path
    cloudFrontUrl = os.environ.get("CLOUD_FRONT_URL")
    print(cloudFrontUrl)

    return "CloudFrontUrl : {}".format(cloudFrontUrl)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
