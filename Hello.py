import streamlit as st
from streamlit.logger import get_logger
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from config import OPENAI_API_KEY

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-4-turbo-preview",temperature=0.7)

LOGGER = get_logger(__name__)
st.write("# Welcome to hello again Push AI")
website = st.text_input("Please enter your Website",help="Enter a website in the pattern of https://www.website.at")

#Fetch website

result = st.button("Start")
if result:

  loader = AsyncHtmlLoader(website)
  docs = loader.load()
  html2text = Html2TextTransformer()
  docs_transformed = html2text.transform_documents(docs)
  docs_transformed[0].page_content[0:10000]
  
