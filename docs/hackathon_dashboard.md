# Hackathon Dashboard

## **Описание задачи**

 Дашборд должен отображать аналитику про модель и бота в целом на основе:
 * логов запросов пользователей
 * базовых метрик, файл [`metrics/`](../metrics.py).
 * (⭐) выдумать метрику для оценки следующей ситуации с моделью: в контекст пришли несколько кусочков документов, но модель начала путать понятия и итоговый ответ получился очень убедительным с точки зрения пользователя, который не знаком с правилами ВШЭ, однако по своей сути неверным :()

## **Основные задачи**
1. **Визуализация пользовательских метрик:**
   - Распределение по кампусам 
    ```python
    campuses = ["Москва", "Нижний Новгород", "Санкт-Петербург", "Пермь"]
    ```

   - Разбивка по уровням образования 
    ```python
    education_levels = ["бакалавриат", "магистратура", "специалитет", "аспирантура"]
    ```

2. **Визуализация метрик вопросов:**
   - Категории вопросов 
    ```python
    question_categories = [
    "Деньги",
    "Учебный процесс",
    "Практическая подготовка",
    "ГИА",
    "Траектории обучения",
    "Английский язык",
    "Цифровые компетенции",
    "Перемещения студентов / Изменения статусов студентов",
    "Онлайн-обучение",
    "Цифровые системы",
    "Обратная связь",
    "Дополнительное образование",
    "Безопасность",
    "Наука",
    "Социальные вопросы",
    "ВУЦ",
    "Общежития",
    "ОВЗ",
    "Внеучебка",
    "Выпускникам",
    "Другое"
    ]
    ```

3. **История чатов:**
   - Анализ частоты повторяющихся вопросов. (близких по смыслу)
   - Подсчет базовых метрик ([`metrics/`](../metrics.py)) и ваших, которые сочтете нужными.

4. **Общая производительность:**
   - Среднее время обработки вопросов. ("Время ответа модели" в логах модели)
   - Частота пустых параметров chat_history (это значит что пользователю понравился ответ модели и он не стал переспрашивать)
   - Частота непустых параметров chat_history (это значит что пользователю не понравился ответ модели и он решил уточнить свой вопрос)

## **Входные данные**

Логи модели выглядят следующим образом:
```json
{
    "Выбранная роль": "Студент",
    "Кампус": "Нижний Новгород",
    "Уровень образования": "Бакалавриат",
    "Категория вопроса": "Учеба",
    "Время ответа модели": 3.294378,
    "user_filters": [
        "Нижний Новгород",
        "бакалавриат"
    ],
    "question_filters": [
        "Учебный процесс",
        "Практическая подготовка"
    ],
    "chat_history": {
        "old_contexts": [
            "Пример текста контекста"
        ],
        "old_questions": [
            "Когда пересдача?"
        ],
        "old_answers": [
            "Пересдача пройдет в сентябре."
        ]
    }
}
```

**Важно:** Иногда `chat_history` может быть пустым

## **Ссылка на Jupyter Notebook**

Для детальной реализации аналитики и дашборда обратитесь к файлу:
[Хакатон_задание_дашборд.ipynb](../notebooks/Hackathon_dashboard.ipynb)



