import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data_file = 'dataset.csv'
data = pd.read_csv(data_file)

# Преобразование данных в формат, совместимый с Hugging Face Datasets
data.columns = ["specification", "user_story"]
dataset = Dataset.from_pandas(data)

# Функция для токенизации данных
def preprocess_function(examples):
    inputs = [spec + " " + tokenizer.eos_token for spec in examples['specification']]
    labels = [story + " " + tokenizer.eos_token for story in examples['user_story']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    labels_tokenized = tokenizer(labels, max_length=512, truncation=True, padding="max_length", return_tensors="pt")['input_ids']
    model_inputs["labels"] = labels_tokenized
    return model_inputs

# Применение токенизации к датасету
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Разделение на обучающую и валидационную выборки
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)

# Определение аргументов для обучения
training_args = TrainingArguments(
    output_dir="./gpt2_user_stories_model",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=200,
)

# Инициализация Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_split['train'],
    eval_dataset=train_test_split['test'],
)

# Запуск обучения
trainer.train()

# Сохранение модели после обучения
trainer.save_model("./gpt2_user_stories_model")
tokenizer.save_pretrained("./gpt2_user_stories_model")

# Функция для генерации юзер-историй на основе требований
def generate_user_story(input_specification):
    input_ids = tokenizer(input_specification + tokenizer.eos_token, return_tensors='pt').input_ids.to(device)
    output = model.generate(input_ids, max_length=60)
    return tokenizer.decode(output[0], skip_special_tokens=True)

