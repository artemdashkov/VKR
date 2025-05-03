from flask import Flask, request, render_template
import re
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
        test_case_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        pattern = r"Test Case:\s*(.+?);?\s*Steps:\s*(.+?);?\s*Expected Result:\s*(.+)$"
        match = re.match(pattern, test_case_text)

        if match:
            ID = "1"
            title = match.group(1).strip()
            steps_raw = match.group(2).strip()
            steps = []
            for step in steps_raw.split("; "):
                step = str(int(steps_raw.split("; ").index(step)) + 1) + "." + " " + step
                steps.append(step)
            expected_result = match.group(3).strip()
            list_of_tests.append({
                    'ID': ID,
                    'Title test case': title,
                    'Steps': steps,
                    'Expected Result': expected_result
                })
        else:
            print("Не удалось распарсить строку")

    return render_template('index.html', list_of_tests=list_of_tests)


if __name__ == '__main__':
    app.run(debug=True)