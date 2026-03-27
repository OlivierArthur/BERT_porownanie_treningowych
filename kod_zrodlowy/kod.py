import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from huggingface_hub import login
from huggingface_hub import notebook_login
notebook_login()


#kolumny muszą być text albo label

def obrabianie_zbioru(ds):
    col_names = ds.column_names

    text_col = next((col for col in col_names if col.lower() in ["text", "email text", "message", "body", "content", "email"]), None)
    label_col = next((col for col in col_names if col.lower() in ["label", "labels", "is_spam", "spam", "target", "class", "email type"]), None)

    if text_col and text_col != "text":
        ds = ds.rename_column(text_col, "text")

    if label_col and label_col != "label":
        ds = ds.rename_column(label_col, "label")

    def format_labels(example):
        val = str(example["label"]).lower().strip()
        if val in ["1", "phishing email", "spam", "phishing", "true"]:
            return {"label": 1}
        else:
            return {"label": 0}

    ds = ds.map(format_labels)
    return ds.select_columns(["text", "label"])



train_data = load_dataset("puyang2025/seven-phishing-email-datasets", split="train")
train_data = obrabianie_zbioru(train_data).shuffle(seed=42)


test_data = load_dataset("zefang-liu/phishing-email-dataset", split="train")
test_data = obrabianie_zbioru(test_data)

print(f"Trenowanie {len(train_data)}")
print(f"Testowanie {len(test_data)}")

train_data = train_data.filter(lambda x: x["text"] is not None and str(x["text"]).strip() != "")
test_data = test_data.filter(lambda x: x["text"] is not None and str(x["text"]).strip() != "")

train_data = train_data.map(lambda x: {"text": str(x["text"])})
test_data = test_data.map(lambda x: {"text": str(x["text"])})

#tokenizacja
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=["text"])
tokenized_test = test_data.map(preprocess_function, batched=True, remove_columns=["text"])


f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"],
        "precision": precision_metric.compute(predictions=predictions, references=labels)["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels)["recall"],
    }


id2label = {0: "ham", 1: "spam"}
label2id = {"ham": 0, "spam": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False

#Musimy obniżyć batch size do 8 przy max_length=512
#chcemy żeby nasz model miał możliwie najlepsze f1
#5 epok to zwykle za dużo jak na bert, mamy early stopping
training_args = TrainingArguments(
    output_dir="./klasyfikatorspamu1",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.05,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_dir='./logs',
    logging_steps=100,
    push_to_hub=True,
    hub_model_id="OliverArt5500/klasyfikatorspamu1",
    hub_strategy="end",
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#wspomniany early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


print("Trwa trenowanie: ")
trainer.train()

print("\n testowanie: ")
metrics = trainer.evaluate()
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")


trainer.save_model("./klasyfikatorspamu1")
trainer.push_to_hub(commit_message=" Aktualizacja modelu klasyfikatorspamu1")

print("koniec")
