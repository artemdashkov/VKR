import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

model_name = "t5-small"

# Инициализация токенизатора и модели
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Загрузка данных
data_file = 'dataset.csv'
data = pd.read_csv(data_file)

# Преобразование данных в формат, совместимый с Hugging Face Datasets
data.columns = ["specification", "user_story"]
dataset = Dataset.from_pandas(data)

# Функция для токенизации данных
def preprocess_function(examples):
    inputs = examples['specification']
    labels = examples['user_story']

    # Токенизация входов и меток
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels_tokenized = tokenizer(labels, max_length=512, truncation=True, padding="max_length")['input_ids']

    # Установка меток
    model_inputs["labels"] = labels_tokenized
    return model_inputs

# Применение токенизации к датасету
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Разделение на обучающую и валидационную выборки
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)

# Определение аргументов для обучения
training_args = TrainingArguments(
    output_dir="content/drive/MyDrive/t5_user_stories_model",  # Директория для сохранения модели
    learning_rate=5e-5,  # Скорость обучения
    per_device_train_batch_size=8,  # Размер батча
    num_train_epochs=5,  # Увеличенное количество эпох
    weight_decay=0.01,  # Регуляризация
    save_strategy="epoch",  # Сохранение модели на каждой эпохе
    logging_steps=200,  # Шаги логирования
    save_total_limit=2,  # Максимальное количество сохраняемых моделей
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
trainer.save_model("./t5_user_stories_model")
tokenizer.save_pretrained("./t5_user_stories_model")