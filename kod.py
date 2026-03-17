import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import mlflow
import dagshub

dagshub.init(repo_owner='OlivierArthur', repo_name='BERT_porownanie_treningowych', mlflow=True)
mlflow.set_experiment("BERT_Spam_porown_zb_treng")

# -o to nadpisane plikow
os.system("kaggle datasets download -d nitishabharathi/email-spam-dataset --unzip -o")

datasety = [
    {"nazwa": "Exp_1_SpamAssassin", "csv_nazwa": "completeSpamAssassin.csv", "text": "Body", "label": "Label"},
    {"nazwa": "Exp_2_Enron",        "csv_nazwa": "enronSpamSubset.csv",      "text": "Body", "label": "Label"},
    {"nazwa": "Exp_3_LingSpam",     "csv_nazwa": "lingSpam.csv",             "text": "Body", "label": "Label"}
]

#klasa konwertująca surowe tokeny i etykiety na tensory wymagane przez bibliotekę PyTorch
class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#inicjalizacja tokenizatora z pre-trenowanego modelu BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    acc = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

for data in datasety:
    print(f"\n START EKSPERYMENTU: {data['nazwa']}")

    #wczytanie
    df = pd.read_csv(data["csv_nazwa"], encoding='utf-8', encoding_errors='replace')

    #kolumny i drop na
    df = df[[data['text'], data['label']]].dropna()

    df = df.drop_duplicates(subset=[data['text']], keep='first') #usuwamy duplikaty

    #ograniczenie zbioru do max 2000 losowych próbek
    #df = df.sample(n=min(2000, len(df)), random_state=42)

    #lebel encoder do wartosci binarnych
    le = LabelEncoder()
    labels = le.fit_transform(df[data['label']])
    texts = df[data['text']].astype(str).tolist()

    #podział na zbiór treningowy (80%) i walidacyjny (20%)
    #train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    #stratified sampling pomoże z imbalansem danych (np. Lingspam mógł się uczyć na samym hamie w teorii)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

    #tokenizacja
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    #Pytorch wymaga takiego czegoś
    train_dataset = SpamDataset(train_encodings, train_labels)
    val_dataset = SpamDataset(val_encodings, val_labels)


    with mlflow.start_run(run_name=(data['nazwa'])+' stratified, zamrożone 5, pełne dane'):

        #pobranie berta i dodanie warstwy do klasyfikacji z 2 labels - 2 klasy
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) #Mozna usunac '#' poniżej żeby trenować tylko warstwę klasyfikacyjną, wypada do tego zwiększyć learning_rate do 2e-3
        #for param in model.bert.parameters():
          #param.requires_grad = False

        #zamrażamy pierwszą warstwę
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

        #Zamrażamy pierwsze 5 warstw, zmienną można zmieniać
        warstwy_do_zamrozenia = 5
        for i in range(warstwy_do_zamrozenia):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False


        training_args = TrainingArguments(
            output_dir=f"./wyniki_{data['nazwa']}",
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            learning_rate=2e-5,
            report_to="mlflow",
            logging_steps=10,
            weight_decay=0.01
        )


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        print(f"trening na danych {data['csv_nazwa']}...")
        trainer.train()

        trainer.evaluate()

        del model
        del trainer
        import gc
        gc.collect() #Wymuszenie odśmiecacza pamięci
        torch.cuda.empty_cache()

        print(f" Zakończono: {data['nazwa']}")

print("\n KONIEC")
