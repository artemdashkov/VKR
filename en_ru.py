from googletrans import Translator # https://pypi.org/project/translators/


# Создаем объект Translator
translator = Translator()

# Текст для перевода
text_to_translate = "Привет, как дела?"

# Переводим текст с русского на английский
translated = translator.translate(text_to_translate, src='ru', dest='en')

# Выводим результат
print(f"Исходный текст: {text_to_translate}")
print(f"Переведенный текст: {translated.text}")

with open('data/us_eng.txt', 'r', encoding='utf-8') as file:
    for x in file.readlines():
        en_line = x.strip()
        translated = translator.translate(en_line, src='en', dest='ru')
        ru_line = translated.text
        ru_line = ru_line.replace('   ', ' ')
        ru_line = ru_line.replace('   ', ' ')
        ru_line = ru_line.replace('​​', ' ')
        ru_line = ru_line.replace('  ', ' ')
        ru_line = ru_line.replace('  ', ' ')
        ru_line = ru_line.replace(' -', '-')
        # print(en_line)
        print(ru_line)