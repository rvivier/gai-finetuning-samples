import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd

import os
import zipfile
import json
import re

import sys
import subprocess
    
subprocess.check_call([sys.executable, "-m", "pip", "install","-r","/opt/ml/processing/input/code/dataset_preparation/requirements.txt"])

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    
    logger.info("Starting preprocessing.")
    
    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    
    #get bucket and key and download the zip file
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])
    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/raw-data.zip"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    #extracting zip file within the container file system
    logger.info("Unzipping downloaded data.")

    with zipfile.ZipFile(fn, 'r') as zip_ref:
        zip_ref.extractall(f"{base_dir}/data/")
    
    #going through the folders
    articles_folder = "News Articles"
    summaries_folder = "Summaries"
    sub_folders = ["business", "entertainment", "politics", "sport", "tech"]

    articles_folders = f"{base_dir}/data/BBC_news_summary/" + articles_folder
    summaries_folder = f"{base_dir}/data/BBC_news_summary/" + summaries_folder
    
    with open('/opt/ml/processing/output/output.jsonl', 'w') as outfile:
        for folder in os.scandir(path = articles_folders):
            counter = 0
            for filename in os.scandir(path = articles_folders + "/" + str(folder.name)):
                if filename.is_file():
                    try:
                        #create article id of the form folder_001
                        id_article = str(folder.name) + "_" + str(filename.name).split(".")[0]

                        #get article content
                        content = ""
                        with open(filename, 'rb') as file:
                            content = file.read()
                        #get article summary
                        summary = ""
                        equivalent_summary_file = summaries_folder + "/" + str(folder.name) + "/" + str(filename.name)
                        with open(equivalent_summary_file, 'rb') as file:
                            summary = file.read()

                        #create json object
                        data = {}
                        data['id'] = id_article
                        data['content'] = content.decode("utf-8")
                        data['summary'] = summary.decode("utf-8")
                        #print(counter)
                        #json_data = json.dumps(data)
                        #print(json_data)
                        #write jsonline
                        json.dump(data, outfile)
                        outfile.write('\n')
                    
                    except UnicodeDecodeError:
                        print(f"skipping:{id_article} due to UnicodeDecodeError")