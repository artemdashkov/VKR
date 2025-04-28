import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Загрузка предобученной модели и токенизатора
model_name = "./t5_user_stories_model"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Функция для генерации юзер-историй на основе требований
def generate_user_story(input_specification):
    input_ids = tokenizer(input_specification + tokenizer.eos_token, return_tensors='pt').input_ids.to(device)
    output = model.generate(input_ids, max_length=60)
    return tokenizer.decode(output[0], skip_special_tokens=True)

result_1 = generate_user_story("Create a test case for the requirement: 'User should be able to subscribe to newsletters.'")

print(result_1)