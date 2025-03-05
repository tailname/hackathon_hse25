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
st.set_page_config(
    page_title='HSE-BOT Dashboard',
    page_icon='✅',
    layout='wide'
)

# Создаём боковое меню
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите дашборд", ["Общий обзор", "ROUGE", "BLEU", "chrF", "BERTScore"])


# Define a function to create a gauge chart
def gauge_chart(value, title, min_value=0, max_value=1):

    st.write(f"**{title}**")
    st.markdown(
        f"""
        <style>
        .stProgress > div > div > div > div {{
            background-color: #d73027; /* Red */
        }}
        .stProgress > div > div > div:nth-child(2) > div {{
            background-color: #f46d43; /* Orange-Red */
        }}
        .stProgress > div > div > div:nth-child(3) > div {{
            background-color: #fdae61; /* Orange */
        }}
        .stProgress > div > div > div:nth-child(4) > div {{
            background-color: #fee08b; /* Yellow */
        }}
        .stProgress > div > div > div:nth-child(5) > div {{
            background-color: #ffffbf; /* Light Yellow */
        }}
        .stProgress > div > div > div:nth-child(6) > div {{
            background-color: #d9ef8b; /* Light Green */
        }}
        .stProgress > div > div > div:nth-child(7) > div {{
            background-color: #a6d96a; /* Green */
        }}
        .stProgress > div > div > div:nth-child(8) > div {{
            background-color: #66bd63; /* Darker Green */
        }}
        .stProgress > div > div > div:nth-child(9) > div {{
            background-color: #1a9850; /* Dark Green */
        }}
        .stProgress > div > div > div:nth-child(10) > div {{
            background-color: #006837; /* Darkest Green */
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Ensure value is within bounds
    value = max(min(value, max_value), min_value)

    progress_value = (value - min_value) / (max_value - min_value)
    st.progress(progress_value)
    st.write(f"Current Value: {value:.4f}")


# Главная страница
if page == "Общий обзор":
    st.title("📊 Общий обзор метрик")
    st.write("Средние значения метрик:")

    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            gauge_chart(df['rouge'].mean(), "ROUGE")
        with col2:
            gauge_chart(df['bleu'].mean(), "BLEU")
        with col3:
            gauge_chart(df['chrf'].mean(), "chrF", max_value=100)
        with col4:
            gauge_chart(df['bertscore'].mean(), "BERTScore")
    else:
        st.warning("No data available!")
    # Используем одну колонку для размещения графика по центру
    col1, col2, col3 = st.columns([1, 6, 1])  # Вторая колонка будет в 6 раз шире остальных

    with col2:
        st.write("ROUGE & BLEU & BERTScore зависимости")
        chart_data = pd.DataFrame({
            "ROUGE": df["rouge"],
            "BLEU": df["bleu"],
            "BERTScore": df["bertscore"]
        })
        st.line_chart(chart_data)


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
