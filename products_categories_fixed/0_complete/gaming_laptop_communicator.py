import sqlite3
import json
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import SQLDatabase
from dotenv import load_dotenv, find_dotenv
import sys
import stat
from langchain_community.llms import HuggingFaceEndpoint
from langchain_anthropic import ChatAnthropic

def get_features_string(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query to select all features
    cursor.execute("SELECT feature_name, feature_value FROM feature_gaming_laptops WHERE feature_value IS NOT NULL")
    
    # Fetch all rows from the executed query
    features = cursor.fetchall()
    
    # Close the database connection
    conn.close()
    
    # Format the features into the desired string format
    features_string = ", ".join([f"{name}={value}" for name, value in features])
    
    return features_string

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#Example usage
db_path = 'product_database_gaming_laptops.db'
features_string = get_features_string(db_path)

conn = sqlite3.connect('product_database_gaming_laptops.db')
c = conn.cursor()
input_db = SQLDatabase.from_uri('sqlite:///product_database_gaming_laptops.db')

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Please note that in the schema, the regular_price is the normal price, while the happy_price is a discounted one.
Table Descriptions:

gaming_laptops Table:
This table stores general information about gaming laptops.
Columns:
id: Unique identifier for each gaming laptop.
url: URL link to the product page.
title: Title or name of the gaming laptop.
warranty: Duration of warranty in months.
regular_price: Normal price of the gaming laptop.
happy_price: Discounted price (if applicable).
category: Category of the gaming laptop (e.g., “gaming_laptops”).

feature_gaming_laptops Table:
This table captures individual features of gaming laptops.
Columns:
id: Unique identifier for each feature.
feature_name: Name of the feature (e.g., “graphics card,” “RAM size”).
feature_value: Value associated with the feature (e.g., “NVIDIA RTX 3080,” “16GB”).

gaming_laptop_feature Table:
This table establishes a many-to-many relationship between gaming laptops and their features.
Columns:
product_id: Foreign key referencing the id column in the gaming_laptops table.
feature_id: Foreign key referencing the id column in the feature_gaming_laptops table.

Example input for questions asking for multiple features: 'What gaming laptops are there that have an NVIDIA graphics card and have at least 8GB of RAM?'
Example output for questions asking for multiple features:
'SELECT gaming_laptops.title, feature_gaming_laptops.feature_name, feature_gaming_laptops.feature_value
FROM gaming_laptops
JOIN gaming_laptop_feature ON gaming_laptops.id = gaming_laptop_feature.product_id
JOIN feature_gaming_laptops ON gaming_laptop_feature.feature_id = feature_gaming_laptops.id
WHERE feature_gaming_laptops.feature_name = 'gpu_model'
  AND feature_gaming_laptops.feature_value LIKE '%NVIDIA%'
  AND gaming_laptops.id IN (
    SELECT glf.product_id
    FROM gaming_laptop_feature glf
    JOIN feature_gaming_laptops fgl ON glf.feature_id = fgl.id
    WHERE fgl.feature_name = 'ram_size_gb' AND fgl.feature_value >= 8
);'

Don't include a feature in the query if the user doesn't ask for it.

You ABSOLUTELY MUST use the ‘LIKE’ operator instead of ‘=’ in the SQL query for columns in the tables that are of TEXT value.
You ABSOLUTELY MUST use the ‘=’ operator instead of ‘LIKE’ in the SQL query for columns in the tables that are of REAL, DOUBLE, or INTEGER value.
Here are all the features a user might ask for:
{features}
Question: {question}
SQL Query:
"""
prompt = ChatPromptTemplate.from_template(template)

def get_schema(db):
    schema = input_db.get_table_info()
    return schema

## For MistralAI
# repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
# llm = HuggingFaceEndpoint(
#     repo_id=repo_id, max_length=128, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN
# )

## For Gemini
# llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

## For Claude Anthropic
# llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")

# sql_chain = (
#     RunnablePassthrough.assign(schema=get_schema)
#     | prompt
#     | llm.bind(stop=["\nSQLResult:"])
#     | StrOutputParser()
# )

# user_question = 'What gaming laptops are there that have an NVIDIA graphics card and have at least 8GB of RAM?'
# smth= sql_chain.invoke({"features": features_string,"question": user_question})
# print(smth)
# output_string = smth.replace("```sql", "").replace("```", "")
# print(output_string)

def generate_sql_query_gaming_laptop_mistral(user_question):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN
    )
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    result = sql_chain.invoke({"features": features_string, "question": user_question})
    return result

def generate_sql_query_gaming_laptop_google(user_question):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    result = sql_chain.invoke({"features": features_string, "question": user_question})
    return result

def generate_sql_query_gaming_laptop_anthropic(user_question):
    llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    result = sql_chain.invoke({"features": features_string, "question": user_question})
    return result