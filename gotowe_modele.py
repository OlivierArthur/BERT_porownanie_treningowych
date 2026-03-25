import os
import pandas as pd
import mlflow
import dagshub
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from google.colab import userdata

os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

dagshub.init(repo_owner='OlivierArthur', repo_name='BERT_porownanie_treningowych', mlflow=True)
mlflow.set_experiment("BERT_Testowanie_3epoki")

device = 0 if torch.cuda.is_available() else -1

df_pom = [
    {"nazwa": "Exp_1_SpamAssassin/checkpoint-1590"},
    {"nazwa": "Exp_2_Enron/checkpoint-2907"},
    {"nazwa": "Exp_3_LingSpam/checkpoint-390"}
]

os.system("kaggle datasets download -d bayes2003/emails-for-spam-or-ham-classification-trec-2007 --unzip -o")

df_raw = pd.read_csv("email_text.csv", encoding='utf-8', encoding_errors='replace')
df_cleaned = df_raw[['text', 'label']].dropna()

texts = df_cleaned['text'].astype(str).tolist()
true_labels = df_cleaned['label'].astype(int).tolist()

for model_info in df_pom:
    model_path = f"./wyniki_{model_info['nazwa']}"

    print(f"\n Test dla: {model_info['nazwa']}")

    with mlflow.start_run(run_name=f"Test_{model_info['nazwa']}_on_TREC"):

        klasyfikator = pipeline(
            task="text-classification",
            model=model_path,
            tokenizer="bert-base-uncased",
            device=device
        )

        batch_size = 32
        predictions_labels = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_results = klasyfikator(batch_texts, truncation=True, max_length=128)
            predictions_labels.extend([1 if r['label'] == 'LABEL_1' else 0 for r in batch_results])

        acc = accuracy_score(true_labels, predictions_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions_labels, average='binary')

        print(f"Wyniki - Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")

        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall})

        del klasyfikator
        torch.cuda.empty_cache()


