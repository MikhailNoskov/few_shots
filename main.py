import os
import yaml
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()
MODEL = os.getenv("MODEL_NAME", "gpt-5")

llm = ChatOpenAI(model=MODEL, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

with open("prompts_config.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

# example_prompt = PromptTemplate.from_template(
#     "Ввод: {input}\nВывод: {output}"
# )
# examples = [
#     {"input": "happy", "output": "sad"},
#     {"input": "tall", "output": "short"}
# ]
example_prompt = PromptTemplate.from_template("Вопрос: {question}\nОтвет: {answer}")
examples = [
    {"question": "Что делать, если опоздал на работу?",
     "answer": "Притворись, что это спецплан компании по тестированию терпения коллег."},
    {"question": "Как победить лень?",
     "answer": "Скажи лени, что завтра — её выходной, и действуй, пока она отдыхает."},
    {"question": "Что делать, если забыл день рождения друга?",
     "answer": "Сделай вид, что это сюрприз для него, и улыбайся, когда он удивлённо морщит лоб."}
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=2  # выбираем 2 ближайших примера
)
# prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     prefix="Дайте антоним каждого слова:",
#     suffix="Ввод: {input}\nВывод:",
#     input_variables=["input"]
# )
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Отвечай на вопросы в шутливо-ироничном стиле, как в примерах:",
    suffix="Вопрос: {question}\nОтвет:",
    input_variables=["question"]
)
# formatted_prompt = prompt.format(input="young")
# response = llm.invoke(formatted_prompt)
question = "Как объяснить начальнику, что проект задерживается?"
formatted_prompt = prompt.format(question=question)
response = llm.invoke(formatted_prompt)
print(response.content) # Вывод: old