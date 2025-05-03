## "Модели машинного обучения для генерации тестовых сценариев на основании технического задания"

![Иллюстрация к проекту](https://github.com/artemdashkov/VKR/start_page.PNG)

**Для решения задачи были применены две модели:**
1. Трансформерная модель `Т5` (t5-small)
2. LLM модель `GPT2`

### Трансформерная модель `Т5`
- `train_t5_2.py`: используется для обучения модели
- `dataset.csv`: датасет для обучения
- `predict_t5_2.py`: используется для генерации тестовых сценариев на основании технического задания
- `./t5_user_stories_model`: директория предобученной модели **(размер: 1,57 Гб!)**

### Результаты работы модели `Т5`:
- Вход:
    - Create a test case for the requirement: 'User should be able to create and manage a shopping list'"
- Выход: 
    - Test Case: Create and Manage Shopping List; 
    - Steps: 
        - Log in; 
        - Navigate to shopping list section; 
        - Click 'Save';  
    - Expected Result: 
        - User successfully creates and manages a shopping list.

### LLM модель `GPT2`
- `train_LLM.py`: используется для обучения модели
- `dataset.csv`: датасет для обучения
- `predict_LLM.py`: используется для генерации тестовых сценариев на основании технического задания 
- `./gpt2_user_stories_model`: директория предобученной модели **(размер: 7,41 Гб)**

### Результаты работы модели:
- Вход:
    - Create a test case for the requirement: 'User should be able to create and manage a shopping list'"
- Выход: 
    - Test Case: Create and Manage Shopping List; 
    - Steps: 
        - Log in; 
        - Navigate to shopping list; 
        - Click 'Save'; 
    - Expected Result: 
        - User successfully creates and manages a shopping list.


