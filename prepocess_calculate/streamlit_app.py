import streamlit as st
import pandas as pd
import json
import os

OUTPUT_FILE = "real_time_results.txt"


# Функция для загрузки данных из файла
def load_metrics():
    if not os.path.exists(OUTPUT_FILE):
        return pd.DataFrame({})

    data_list = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                data_list.append({
                    "rouge": data.get("context_recall")[0],
                    "bleu": data.get("context_precision")[0],
                    "chrf": data.get("answer_correctness_literal")[0],
                    "bertscore": data.get("answer_correctness_neural")[0][0]
                })
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(data_list)


# Загружаем данные
df = load_metrics()

# Создаём боковое меню
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите дашборд", ["Общий обзор", "ROUGE", "BLEU", "chrF", "BERTScore"])

# Главная страница
if page == "Общий обзор":
    st.title("📊 Общий обзор метрик")
    st.write("Средние значения метрик:")
    st.write(df.mean())

    # Визуализация в виде столбчатой диаграммы
    st.bar_chart(df.mean())

# Дашборды по метрикам
else:
    st.title(f"📈 Дашборд: {page}")

    # Выбор диапазона значений
    min_val, max_val = st.slider(f"Выберите диапазон значений для {page}",
                                 float(df[page.lower()].min()),
                                 float(df[page.lower()].max()),
                                 (float(df[page.lower()].min()), float(df[page.lower()].max())))

    # Фильтрация данных
    filtered_df = df[(df[page.lower()] >= min_val) & (df[page.lower()] <= max_val)]

    # Отображение графика
    st.line_chart(filtered_df[page.lower()])
    chart1, chart2 = st.beta_columns(2)

    with chart1:
        chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
        st.line_chart(chart_data)

    with chart2:
        chart_data = pd.DataFrame(np.random.randn(2000, 3),columns=['a', 'b', 'c'])
        st.line_chart(chart_data)

