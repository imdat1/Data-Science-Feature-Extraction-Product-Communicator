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
    cursor.execute("SELECT feature_name, feature_value FROM feature_mobile_phones WHERE feature_value IS NOT NULL")
    
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

# Example usage
db_path = 'product_database_mobile_phones.db'
features_string = get_features_string(db_path) 

conn = sqlite3.connect('product_database_mobile_phones.db')
c = conn.cursor()
input_db = SQLDatabase.from_uri('sqlite:///product_database_mobile_phones.db')

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Please note that in the schema, the regular_price is the normal price, while the happy_price is a discounted one.
Table Descriptions:
mobile_phones Table:
This table stores general information about mobile phones.
Columns:
id: Unique identifier for each mobile phone.
url: URL link to the product page.
title: Title or name of the mobile phone.
warranty: Duration of warranty in months.
regular_price: Normal price of the mobile phone.
happy_price: Discounted price (if applicable).
category: Category of the mobile phone (e.g., “gaming_laptops”).

feature_mobile_phones Table:
This table captures individual features of mobile phones.
Columns:
id: Unique identifier for each feature.
feature_name: Name of the feature (e.g., “camera resolution,” “battery capacity”).
feature_value: Value associated with the feature (e.g., “12 MP,” “4000 mAh”).

mobile_phone_feature Table:
This table establishes a many-to-many relationship between mobile phones and their features.
Columns:
product_id: Foreign key referencing the id column in the mobile_phones table.
feature_id: Foreign key referencing the id column in the feature_mobile_phones table.

Example input for questions asking for multiple features: 'What phones are there that have a screen size of at least 6 inches and an OLED screen?'
Example output for questions asking for multiple features:
'SELECT mobile_phones.title, screen_size.feature_value AS screen_size, screen_type.feature_value AS screen_type
FROM mobile_phones
JOIN mobile_phone_feature AS mpf1 ON mobile_phones.id = mpf1.product_id
JOIN feature_mobile_phones AS screen_size ON mpf1.feature_id = screen_size.id
JOIN mobile_phone_feature AS mpf2 ON mobile_phones.id = mpf2.product_id
JOIN feature_mobile_phones AS screen_type ON mpf2.feature_id = screen_type.id
WHERE screen_size.feature_name = 'screen_size_inches'
AND CAST(screen_size.feature_value AS UNSIGNED) >= 6
AND screen_type.feature_name = 'screen_type'
AND screen_type.feature_value LIKE '%OLED%';'
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

# user_question = 'Can you give me all the phones with a camera that is at least 50MP and the size is at least 6 inches?'
# smth= sql_chain.invoke({"features": features_string,"question": user_question})
# print(smth)
# output_string = smth.replace("```sql", "").replace("```", "")
# print(output_string)

def generate_sql_query_mobile_phone_mistral(user_question):
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

def generate_sql_query_mobile_phone_google(user_question):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    result = sql_chain.invoke({"features": features_string, "question": user_question})
    return result

def generate_sql_query_mobile_phone_anthropic(user_question):
    llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    result = sql_chain.invoke({"features": features_string, "question": user_question})
    return result
