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
    list_of_tests = []

    if request.method == 'POST':
        user_story = request.form['user_story']
        # Токенизация и генерация
        inputs = tokenizer(
            user_story,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
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
        test_case_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Просто делим ответ на 3 строки
        lines = test_case_text.split('\n')
        title = lines[0] if len(lines) > 0 else "Title"
        steps = lines[1] if len(lines) > 1 else "Steps"
        expected = lines[2] if len(lines) > 2 else "Expected Result"

        list_of_tests.append({
            'ID': 1,
            'Title test case': title,
            'Steps': steps,
            'Expected Result': expected
        })

    return render_template('index.html', list_of_tests=list_of_tests)

if __name__ == '__main__':
    app.run(debug=True)