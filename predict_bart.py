import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Пути к сохраненной модели и токенизатору
# model_path = '/content/drive/MyDrive/bart_test_case_saved'  # for notebook
model_path = './content/drive/MyDrive/bart_test_case_saved'

# Загружаем модель и токенизатор
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Вводное требование
input_text = "Requirement: 'User must be able to register with a valid email and a strong password'. Create a detailed test case."

# Токенизация входных данных
inputs = tokenizer(
    input_text,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Генерация
outputs = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_new_tokens=150,  # можно увеличить для длинных ответов
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

# Расшифровка и вывод
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Input:\n", input_text)
print("Output:\n", generated_text)

