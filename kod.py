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
        "nazwa": "Exp_1_SpamAssassin",
        "csv_nazwa": "completeSpamAssassin.csv",
        "text": "Body",
        "label": "Label"
    },
    {
        "nazwa": "Exp_2_Enron",
        "csv_nazwa": "enronSpamSubset.csv",
        "text": "Body",
        "label": "Label"
    },
    {
        "nazwa": "Exp_3_LingSpam",
        "csv_nazwa": "lingSpam.csv",
        "text": "Body",
        "label": "Label"
    }
]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

for data in datasety:
  df = pd.read_csv(data["csv_nazwa"])
  df = df[[data['text'], data['label']]].dropna()
  df = df.sample(n=min(2000, len(df)), random_state=42)

  le = LabelEncoder()
  labels = le.fit_transform(df[config['label_col']])
  texts = df[config['text_col']].astype(str).tolist()
    
  train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

  train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='tf')
  val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='tf')
    
  train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).shuffle(1000).batch(16)
  val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(16)

  with mlflow.start_run(run_name=config['name']):
    mlflow.tensorflow.autolog()
        
      model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
      optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
      model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
      print(f"Trenowanie modelu na pliku {config['csv_name']}...")
      model.fit(train_dataset, validation_data=val_dataset, epochs=3)
  tf.keras.backend.clear_session()
    del model
    print(f" Skończone {config['name']} ")


