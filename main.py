import pandas as pd
import numpy as np
from openai import OpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import ast


# Функция загрузки датасета
def load_embeddings_to_dataframe(filepath):
    """
    Загружает данные эмбеддингов из CSV файла в pandas DataFrame

    Args:
        filepath (str): Путь к файлу с данными

    Returns:
        pd.DataFrame: DataFrame с загруженными данными
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Загружено {len(df)} записей из файла {filepath}")
        print(f"Колонки: {df.columns.tolist()}")
        print(f"\nПервые 3 записи:")
        print(df.head(3))
        return df
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных: {e}")
        return None


# Загрузка данных
df = load_embeddings_to_dataframe("arxiv_embeddings202505211515.csv")

# Просмотр информации о датасете
# print(f"Размер датасета: {df.shape}")
# print(f"\nТипы данных:\n{df.dtypes}")
# print(f"\nПроверка пропущенных значений:\n{df.isnull().sum()}")
#

# Подготовка документов в формате LangChain
def prepare_documents(df, limit=1000):
    """
    Преобразует DataFrame в список документов LangChain

    Args:
        df: DataFrame с данными статей
        limit: Количество документов для обработки

    Returns:
        List[Document]: Список документов
    """
    documents = []


    df_subset = df.head(limit)

    for idx, row in df_subset.iterrows():
        page_content = ""
        # Адаптируйте под структуру вашего датасета
        if 'title' in row:
            page_content += f"Название: {row['title']}\n"
        if 'abstract' in row:
            page_content += f"Аннотация: {row['abstract']}\n"
        if 'authors' in row:
            page_content += f"Авторы: {row['authors']}\n"

        # Метаданные
        metadata = {}
        if 'main_category' in row:
            metadata['category'] = row['main_category']
        if 'year' in row:
            metadata['date'] = row['year']
        if 'article_id' in row:
            metadata['arxiv_id'] = row['article_id']

        documents.append(Document(
            page_content=page_content,
            metadata=metadata
        ))

    print(f"Подготовлено {len(documents)} документов")
    return documents


# Подготовка документов
documents = prepare_documents(df, limit=1000)

# Просмотр примера документа
print("Пример документа:")
print(f"Содержимое: {documents[0].page_content[:200]}...")
print(f"Метаданные: {documents[0].metadata}")

# Конфигурация OpenAI клиента для получения эмбеддингов
client = OpenAI(
    api_key="ZmJhMjUwZTItMDg0ZC00N2E3LWIyNDktYjA4MTQyZGFmMGE4.97f6d089a16317c3aa93b365eda739a8",
    base_url="https://foundation-models.api.cloud.ru/v1"
)


def get_embedding(text: str, model="BAAI/bge-m3") -> list:
    """Получает эмбеддинг текста"""
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


# Тестирование функции эмбеддингов
test_embedding = get_embedding("Тестовый запрос")
print(f"Размерность эмбеддинга: {len(test_embedding)}")


# Создание кастомного класса эмбеддингов для GigaChain
from langchain_core.embeddings import Embeddings
from typing import List


class CustomEmbeddings(Embeddings):
    """Кастомный класс эмбеддингов для работы с API"""

    def __init__(self, client, model="BAAI/bge-m3"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов для списка документов"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Получение эмбеддинга для запроса"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding


# Создание экземпляра эмбеддингов
embeddings = CustomEmbeddings(client)

# Создание векторного хранилища ChromaDB
"""
print("Создание векторного хранилища...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="arxiv_papers",
    persist_directory="./chroma_db"
    )
print("Векторное хранилище успешно создано!")
"""
print("Загрузка существующего векторного хранилища...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="arxiv_papers"
)
print("Хранилище загружено из диска.")

# Проверка работы хранилища
# test_query = "искусственный интеллект и роботы"
# results = vectorstore.similarity_search(test_query, k=3)
# print(f"\nРезультаты поиска по запросу '{test_query}':")
# for i, doc in enumerate(results, 1):
#     print(f"\n{i}. {doc.page_content[:200]}...")
#     print(f"   Метаданные: {doc.metadata}")
#
# Эксперимент: Разные значения k
# print("\n" + "="*80)
# print("ЭКСПЕРИМЕНТ 1: Разные значения k")
# for k in [1, 2, 3]:
#     print(f"\nПоиск с k={k} для запроса 'искусственный интеллект и роботы':")
#     results = vectorstore.similarity_search("искусственный интеллект и роботы", k=k)
#     for i, doc in enumerate(results, 1):
#         print(f"{i}. {doc.page_content[:150]}...")

# Эксперимент: Поиск по разным темам
# print("\n" + "="*80)
# print("ЭКСПЕРИМЕНТ 2: Поиск по разным темам")
# queries = [
#     "нейронные сети",
#     "мультиагентные системы",
#     "программное обеспечение"
# ]

# for query in queries:
#     print(f"\nЗапрос: '{query}'")
#     results = vectorstore.similarity_search(query, k=2)
#     for i, doc in enumerate(results, 1):
#         print(f"{i}. {doc.page_content[:300]}...")
#         print(f"   Метаданные: {doc.metadata}")



# Эксперимент: Поиск с оценками схожести
# print("\n" + "="*80)
# print("ЭКСПЕРИМЕНТ 3: Поиск с оценками схожести")
# query = "машинное обучение"
# results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
#
# print(f"Результаты для запроса '{query}':")
# for i, (doc, score) in enumerate(results_with_scores, 1):
#     # В Chroma меньший score = выше схожесть (косинусное расстояние)
#     print(f"\n{i}. Схожесть: {score:.4f}")
#     print(f"   Название: {doc.page_content.split('Аннотация:')[0][9:100]}...")
#     print(f"   Метаданные: {doc.metadata}")

# Создание ретривера с поиском по схожести
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Возвращать топ-5 документов
)

# Тестирование ретривера
# query = "криптография"
# retrieved_docs = retriever.invoke(query)
#
# print(f"Найдено документов: {len(retrieved_docs)}")
# for i, doc in enumerate(retrieved_docs, 1):
#     print(f"\nДокумент {i}:")
#     print(doc.page_content[:150] + "...")

# MMR балансирует между релевантностью и разнообразием результатов
# retriever_mmr = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 5,
#         "fetch_k": 50,  # Количество документов для первичной выборки
#         "lambda_mult": 0  # Баланс между релевантностью (1.0) и разнообразием (0.0)
#     }
# )
# Эксперимент: Сравнение результатов
# Сравнение результатов
# query = "криптография"
# docs_similarity = retriever.invoke(query)
# docs_mmr = retriever_mmr.invoke(query)
#
# print("Сравнение результатов поиска:")
# print("\nSimilarity Search")
# for i, doc in enumerate(docs_similarity[:3], 1):
#     print(f"{i}. {doc.page_content[:100]}...")
#
# print("\nMMR Search")
# for i, doc in enumerate(docs_mmr[:3], 1):
#     print(f"{i}. {doc.page_content[:100]}...")


# Создание ретривера с порогом схожести
# Ретривер с фильтрацией по оценке схожести
# Создание ретривера с порогом схожести через метод as_retriever
# retriever_threshold = vectorstore.as_retriever(
#     search_type="similarity_score_threshold",  # Тип поиска: с порогом схожести
#     search_kwargs={
#         "score_threshold": 1,  # Минимальная оценка схожести (чем меньше, тем более похожи)
#         "k": 10  # Максимальное количество возвращаемых результатов
#     }
# )

# Создание ретривера с порогом схожести
# retriever_threshold = vectorstore.as_retriever(
#     search_type="similarity_score_threshold",  # Тип поиска - с порогом схожести
#     search_kwargs={
#         "score_threshold": 1,  # Минимальная оценка схожести (чем меньше значение, тем выше схожесть)
#         "k": 10  # Максимальное количество возвращаемых результатов
#     }
# )

# Вывод заголовка для раздела поиска с порогом схожести
# print("\nSimilarity Search with Score Threshold")

# Определение поискового запроса для поиска документов
# query = "глубокое обучение для обработки изображений"

# Установка порогового значения для оценки схожести
# threshold = 1

# Комментарий о том, что ChromaDB не поддерживает параметр score_threshold через метод as_retriever
# поэтому используется прямой вызов similarity_search_with_score
# results_with_scores = vectorstore.similarity_search_with_score(
#     query,  # Текст запроса для поиска
#     k=10  # Количество возвращаемых результатов (до фильтрации)
# )

# Фильтрация результатов поиска по установленному порогу схожести
# Оставляем только те документы, оценка схожести которых меньше или равна порогу
# filtered_results = [
#     (doc, score) for doc, score in results_with_scores if score <= threshold
# ]

# Вывод информации о запросе и количестве найденных документов
# print(f"Запрос: '{query}' (порог схожести ≤ {threshold})")
# print(f"Найдено документов после фильтрации: {len(filtered_results)}")

# Перебор отфильтрованных результатов с выводом детальной информации
# for i, (doc, score) in enumerate(filtered_results, 1):
#     # Извлечение заголовка из первой строки содержимого документа
#     title = doc.page_content.split('\n')[0]

#     # Вывод информации о документе: порядковый номер, оценка схожести, заголовок
#     print(f"{i}. [Схожесть: {score:.4f}] {title}")

#     # Вывод метаданных документа (информация о источнике, автор, дата и т.д.)
#     print(f"   Метаданные: {doc.metadata}")

# Эксперимент: Влияние параметра lambda_mult на MMR-поиск
# print("\n" + "="*80)
# print("ЭКСПЕРИМЕНТ С lambda_mult НА ЗАПРОСЕ: 'генерация текстов'")
# query = "генерация текстов"
#
# for lam in [1.0, 0.5, 0.0]:
#     print(f"\nlambda_mult = {lam}")
#     retriever_test = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 3,
#             "fetch_k": 50,
#             "lambda_mult": lam
#         }
#     )
#     docs = retriever_test.invoke(query)
#
#     for i, doc in enumerate(docs, 1):
#         title = doc.page_content.split('\n')[0]  # Берём только название
#         category = doc.metadata.get('category', 'unknown')
#         print(f"{i}. [{category}] {title[:100]}...")
# print("\n" + "="*80)

# Эксперимент: Сравнение времени работы методов поиска
# print("СРАВНЕНИЕ ВРЕМЕНИ РАБОТЫ")
# import time
# import random
#
# query = "глубокое обучение для обработки изображений"
# k = 5
# n_trials = 20
#
# # Прогревочный запуск (игнорируем результат)
# for _ in range(2):
#     vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k}).invoke(query)
#     vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 15, "lambda_mult": 0.5}).invoke(query)
#
# def measure_time(search_type, kwargs):
#     times = []
#     for _ in range(n_trials):
#         # Случайный порядок запусков
#         if random.random() > 0.5:
#             retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=kwargs)
#             start = time.time()
#             retriever.invoke(query)
#             elapsed = time.time() - start
#             times.append(elapsed)
#         else:
#             # Чередуем с другим методом для сброса кэша
#             dummy_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
#             dummy_retriever.invoke("нейтральный запрос")
#     return sum(times) / len(times)
#
# # Замеряем независимо
# time_sim = measure_time("similarity", {"k": k})
# time_mmr = measure_time("mmr", {"k": k, "fetch_k": 15, "lambda_mult": 0.5})
#
# print(f"\nЗапрос: '{query}' | k = {k} | Попыток: {n_trials} (с перемешиванием порядка)")
# print(f"Similarity Search: {time_sim:.4f} сек (в среднем)")
# print(f"MMR Search:        {time_mmr:.4f} сек (в среднем)")
# print(f"MMR медленнее в {time_mmr / time_sim:.1f} раз")
# Задание 4: Разработка RAG-системы

# Настройка языковой модели GigaChat
llm = GigaChat(
    credentials="MDE5OWM0YWYtOTM2My03YmU5LWJmNDQtMmU5ZDYxNDkwMDMzOmVlM2UwZWQyLWRjODUtNGUwMy1hZDIwLWM1N2VkY2U3MmY2Nw==",
    scope="GIGACHAT_API_B2B",
    model="GigaChat-Pro",
    verify_ssl_certs=False,
    timeout=30
)

# Тест языковой модели
test_response = llm.invoke("Привет! Ответь кратко: что такое машинное обучение?")
print(f"Ответ модели: {test_response.content}")


# Промпт-шаблон для RAG-системы
prompt_template = """Ты -- научный ассистент, специализирующийся на анализе научных статей.
Твоя задача -- отвечать на вопросы пользователя, основываясь ТОЛЬКО на предоставленном контексте из научных статей ArXiv.

Правила:
1. Используй только информацию из контекста ниже
2. Если в контексте нет информации для ответа, честно скажи об этом
3. ВСЕГДА указывай полные источники в формате: (Документ N: arxiv_id=..., категория=...).
4. Отвечай на русском языке, четко и структурированно
5. Если вопрос касается технических деталей, будь точным

Контекст из научных статей:
{context}

Вопрос пользователя: {question}

Ответ:"""

prompt = ChatPromptTemplate.from_template(prompt_template)


# Функция форматирования контекста
def format_docs(docs):
    """
    Форматирует список документов в единую строку контекста

    Args:
        docs: Список документов Document

    Returns:
        str: Форматированный контекст
    """
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Документ {i}]")
        context_parts.append(doc.page_content)
        if doc.metadata:
            context_parts.append(f"Метаданные: {doc.metadata}")
        context_parts.append("")  # Пустая строка для разделения
    return "\n".join(context_parts)


# Тест форматирования
test_docs = retriever.invoke("нейронные сети")
formatted_context = format_docs(test_docs[:2])
print("Пример форматированного контекста:")
print(formatted_context[:500] + "...")

# RAG-цепочка
from langchain_core.runnables import RunnablePassthrough

# Создание RAG-цепочки
rag_chain = (
    {
        "context": retriever | format_docs,  # Извлекаем и форматируем документы
        "question": RunnablePassthrough()  # Передаем вопрос как есть
    }
    | prompt  # Формируем промпт
    | llm  # Отправляем в языковую модель
)

# Тестирование RAG-системы
questions = [
    "Какие методы используются для сегментации медицинских изображений?",
    "Как трансформеры применяются в компьютерном зрении?",
    "Какие подходы к обучению без учителя (unsupervised learning) описаны в статьях?",
    "Как обеспечивается безопасность мультимодальных моделей от атак через изображения?",
    "Какие методы аугментации данных используются в глубоком обучении?"
]


print("Тестирование RAG-системы")
for i, question in enumerate(questions, 1):
    print(f"Вопрос {i}: {question}")
    print("-" * 80)

    try:
        response = rag_chain.invoke(question)
        print(f"Ответ: {response.content}\n")
    except Exception as e:
        print(f"Ошибка: {e}\n")


# Интерактивная система вопросов-ответов
def interactive_rag_qa():
    """Интерактивная система вопросов-ответов"""
    print("Интерактивная RAG-система для научных статей ArXiv")
    print("Введите 'выход' для завершения\n")

    while True:
        question = input("Ваш вопрос: ").strip()

        if question.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break

        if not question:
            continue

        try:
            # Получаем релевантные документы
            docs = retriever.invoke(question)
            print(f"\n Найдено релевантных документов: {len(docs)}")

            # Генерируем ответ
            response = rag_chain.invoke(question)
            print(f"\n Ответ:\n{response.content}\n")

            # Показываем источники
            show_sources = input("Показать источники? (да/нет): ").strip().lower()
            if show_sources in ['да', 'yes', 'y', 'д']:
                print("\n Источники:")
                for i, doc in enumerate(docs[:3], 1):
                    print(f"\n{i}. {doc.page_content[:200]}...")
                    print(f"   Метаданные: {doc.metadata}")

            print("\n" + "=" * 80 + "\n")

        except Exception as e:
            print(f" Ошибка: {e}\n")


interactive_rag_qa()
