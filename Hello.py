import streamlit as st
from streamlit.logger import get_logger
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic


llm = ChatOpenAI(openai_api_key=st.secrets.openai_api_key,model="gpt-4-turbo-preview",temperature=0.7)
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
  chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Ein Nutzer stellt dir extrahierte Texte einer Webseite eines Unterhehmens zur Verfügung. Deine Aufgabe ist es folgende Informationen herauszufinden:"
        +"#Name des Unternehmens:# (Wähle dabei den Namen den das Unternehmen in seiner Kommunikation verwendet)\n" 
        +"#Liste angebotener Dienstleistungen und Produkte:# (Wähle dabei maximal 5 Überkategorien\n"
        +"#Vorteile und USPS:# (Wähle dabei die relevantesten USPs. Maximal 4)\n"+
        "#Tonalität und Stil:# (Kommuniziert das Unternehmen per DU mit Kunden, wie förmlich oder locker ist der Stil?"),
        ("human", "{websiteText}")
    ]
)

  messages = chat_template.format_messages(
  websiteText=docs_transformed)


  companyData = llm.invoke(messages).content
  v_occasion = "Aktion oder Anlass: kein spezieller Anlass"
  v_goal ="Ziel der Nachricht: Kunden zurück ins Geschäft holen\n"

  chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du bist ein Kommunikationsspezialist und schreibst 3 Vorschläge einer maßgeschneiderte Push Nachricht auf Basis der vom Nutzer bereitgestellten Unternehmensdetails. Beachte die Details um eine möglichst effektive, wahrheitsgemäße Nachricht zugeschnitten auf das Unternehmen zu verfassen. Jede Push Nachricht soll aus einem title mit maximal 70 Zeichen und einer description mit maximal 150 Zeichen bestehen. Vermeide Angebote, Vorteile oder Aktionen vorzuschlagen die nicht aus den angegebenen Inhalten klar und deutlich hervorgehen. \n\nAusgabeformat:\nBegründe jeden deiner Vorschläge mit einem reasoning mit maximal 150 zeichen. Die Ausabe muss in einem JSON Objekt 'pushmessages' mit den 3 Vorschlägen als Objekten. Jedes Objekt enthält eine ID (1-3) title, description und reasoning."),
        ("human", "{companyDetails}" 
        +  "{occasion}"
        +  "{goal}"
        )
    ]
  )

  messages = chat_template.format_messages(
  companyDetails=companyData,
  occasion=v_occasion,
  goal=v_goal)

  proposedMessages = llm.invoke(messages)
  st.markdown(proposedMessages.content)
  
