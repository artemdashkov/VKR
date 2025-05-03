from flask import Flask, request, render_template
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

# Путь к модели
model_path = './content/drive/MyDrive/bart_test_case_saved'

# Загружаем модель и токенизатор один раз при запуске приложения
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    test_case = ""
    user_story = ""
    if request.method == 'POST':
        user_story = request.form['user_story']
        # Токенизация входных данных
        inputs = tokenizer(
            user_story,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Генерация тест-кейса
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        # Расшифровка
        test_case = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return render_template('index.html', test_case=test_case, user_story=user_story)

if __name__ == '__main__':
    app.run(debug=True)