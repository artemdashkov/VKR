from transformers import T5Tokenizer, T5ForConditionalGeneration  

# Загрузка предобученной модели и токенизатора
# model_name = "/content/drive/MyDrive/t5_user_stories_model"  # for notebook
model_name = "./content/drive/MyDrive/t5_user_stories_model"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Пример входного текста
input_text = "Create a test case for the requirement: 'User should be able to create and manage a shopping list'"

# Токенизация входного текста
input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

# Генерация текста
output_ids = model.generate(input_ids, max_length=100, num_beams=4, do_sample=True, temperature=1.0)

# Декодирование результата
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Intput:", input_text)
print("Output:", output_text)