import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import mlflow
import dagshub

dagshub.init(repo_owner='OlivierArthur', repo_name='BERT_porownanie_porownanie', mlflow=True)
mlflow.set_experiment("BERT_Spam_porown_zb_treng")

os.system("kaggle datasets download -d nitishabharathi/email-spam-dataset --unzip -o")

datasety = [
    {
        "name": "Exp_1_SpamAssassin",
        "csv_name": "completeSpamAssassin.csv",
        "text_col": "Body",      
        "label_col": "Label"
    },
    {
        "name": "Exp_2_Enron",
        "csv_name": "enronSpamSubset.csv",
        "text_col": "Body",
        "label_col": "Label"
    },
    {
        "name": "Exp_3_LingSpam",
        "csv_name": "lingSpam.csv",
        "text_col": "Body",
        "label_col": "Label"
    }
]
