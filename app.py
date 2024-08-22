import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
# Ensure the necessary import
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

# Load environment variables
load_dotenv()

# Setting up the page configuration with title and icon
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="✨")

# Setting up the title of the app
st.header("Chat-Mate...Conversational Analytics Chatbot📝")

# Retrieve the Groq API key from the .env file
api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is provided
if not api_key:
    st.error("Please set the Groq API key in the .env file.")
    st.stop()

# Initialize the Groq LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

# Function to configure SQLite database
@st.cache_resource(ttl="2h")
def configure_db():
    dbfilepath = (Path(__file__).parent / "analytics_db").absolute()
    creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
    return SQLDatabase(create_engine("sqlite:///", creator=creator))

# Configure DB
db = configure_db()

# SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Creating an agent with SQL DB and Groq LLM
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True  # Added to handle parsing errors
)

# Input for user query
question = st.text_input("Enter your question:", key="input")
submit = st.button("Submit")

if submit:
    if question:
        st.subheader("Result:")
        with st.spinner("Generating SQL query..."):
            try:
                # Run the agent and get the response
                response = agent.run(question)
                
                # Display the raw response for debugging
                st.write("Raw Response:", response)

                # Extract and display the SQL query
                # print("Debug")
                print(response, response.upper())
                sql_query = None
                if "SELECT" in response.upper():
                    sql_query = response.strip()
                    st.subheader("Generated SQL Query:")
                    st.code(sql_query, language='sql')
                # else:
                #     st.error("Failed to generate a valid SQL query.")
            except ValueError as ve:
                st.error(f"An error occurred: {str(ve)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
    else:
        st.warning("Please enter a question.")
