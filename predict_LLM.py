from transformers import GPT2LMHeadModel, GPT2Tokenizer

# model_path = "/content/drive/MyDrive/gpt2_finetuned" # for notebook
model_path = "./content/drive/MyDrive/gpt2_finetuned"

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Входной текст
input_text = (
    "Requirement: 'User should be able to create and manage a shopping list'.\n"
    "Create a detailed test case for this requirement.\n"
    "Test case:\n"
)

# input_text = (
#     "Generate a detailed test case for the requirement:\n"
#     "'User should be able to create and manage a shopping list'.\n"
#     "Test case:\n"
# )

# Токенизация с вниманием и маской
inputs = tokenizer.encode_plus(
    input_text,
    return_tensors='pt',
    max_length=512,
    padding='max_length',
    truncation=True
)

# Установка pad_token_id, если его нет
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Генерация с ограничением только новых токенов
outputs = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_new_tokens=200,  # увеличиваем, чтобы дать больше простора для ответа
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
    top_k=50,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input:", input_text)
print("Output:", generated_text)