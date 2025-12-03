import os
import yaml
import sys
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel


__import__('pysqlite3')

load_dotenv()
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


class DataSchema(BaseModel):
    example_prompt: str
    examples: list[dict]
    prefix: str
    suffix: str
    input_variables: list[str]

MODEL = os.getenv("MODEL_NAME", "gpt-5")

llm = ChatOpenAI(model=MODEL, temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

with open("prompts_config.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

prompt_data = DataSchema(**data)

example_prompt = PromptTemplate.from_template(prompt_data.example_prompt)
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=prompt_data.examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=2
)
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prompt_data.prefix,
    suffix=prompt_data.suffix,
    input_variables=prompt_data.input_variables,
)
question = input("Вопрос: ")
formatted_prompt = prompt.format(question=question)
response = llm.invoke(formatted_prompt)
print(response.content)