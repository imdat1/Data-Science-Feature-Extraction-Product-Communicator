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
    cursor.execute("SELECT feature_name, feature_value FROM feature_mouses WHERE feature_value IS NOT NULL")
    
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
db_path = 'product_database_mouses.db'
features_string = get_features_string(db_path) 

conn = sqlite3.connect('product_database_mobile_phones.db')
c = conn.cursor()
input_db = SQLDatabase.from_uri('sqlite:///product_database_mobile_phones.db')

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Please note that in the schema, the regular_price is the normal price, while the happy_price is a discounted one.
Table Descriptions:
mouses Table:
id: An integer representing the unique identifier for each mouse product.
url: A text field to store the URL of the product.
title: A text field for the product title.
warranty: An integer field representing the warranty duration (in months).
regular_price: A real number field for the regular price of the mouse.
happy_price: A real number field for the discounted price (if available).
category: A text field to categorize the mouse product (e.g., ‘Gaming’, ‘Office’, etc.).

feature_mouses Table:
id: An integer representing the unique identifier for each feature.
feature_name: A text field describing a specific feature (e.g., ‘Wireless’, ‘Ergonomic’, etc.).
feature_value: A text field containing the value of the feature (e.g., ‘Yes’, ‘No’, ‘1200 DPI’, etc.).

mouse_feature Table:
product_id: An integer representing the foreign key reference to the id field in the Mouses Table.
feature_id: An integer representing the foreign key reference to the id field in the Feature Mouses Table.

Example input for questions asking for multiple features: 'What OLED TVs are there that have a display bigger than 60 inches?'
Example output for questions asking for multiple features:
'SELECT tvs.title, screen_size.feature_value, screen_type.feature_value 
FROM tvs
JOIN tv_feature AS tvf1 ON tvs.id = tvf1.product_id
JOIN feature_tvs AS screen_size ON tvf1.feature_id = screen_size.id
JOIN tv_feature AS tvf2 ON tvs.id = tvf2.product_id
JOIN feature_tvs AS screen_type ON tvf2.feature_id = screen_type.id
WHERE screen_size.feature_name = 'screen_size_inches'
AND CAST(screen_size.feature_value AS INTEGER) >= 60
AND screen_type.feature_name = 'panel_type'
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

# ## For MistralAI
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

# user_question = 'What wireless mouses are there that weight less than 80 grams?'
# smth= sql_chain.invoke({"features": features_string,"question": user_question})
# print(smth)
# output_string = smth.replace("```sql", "").replace("```", "")
# print(output_string)

def generate_sql_query_mouse_mistral(user_question):
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

def generate_sql_query_mouse_google(user_question):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    result = sql_chain.invoke({"features": features_string, "question": user_question})
    return result

def generate_sql_query_mouse_anthropic(user_question):
    llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    result = sql_chain.invoke({"features": features_string, "question": user_question})
    return result
