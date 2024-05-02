import streamlit as st
import time 
from streamlit.logger import get_logger
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks import get_openai_callback
from langchain_groq import ChatGroq



LOGGER = get_logger(__name__)
st.write("# Welcome to hello again Push AI")
website = st.text_input("Bitte gib deine Webseite ein (inkl. https://)",help="Enter a website in the pattern of https://www.website.at")
inputOccasion = st.text_input("Aktion oder Anlass der Nachricht",value="kein spezifischer Anlass")
inputGoal = st.text_input("Ziel der Nachricht",value="Kunden zurück ins Geschäft holen")

modelOption = st.selectbox(
   "Model",
   ("gpt-4-turbo","claude-3-opus-20240229","claude-3-sonnet-20240229","claude-3-haiku-20240307","llama3-70b-8192"),
   index=0,
   placeholder="Select your model",
)

if "gpt-4-turbo" in modelOption: 
    llm = ChatOpenAI(openai_api_key=st.secrets.openai_api_key,model=modelOption,temperature=0.7)
elif "llama3-70b-8192" in modelOption:
    llm = ChatGroq(groq_api_key=st.secrets.groq_api_key, model_name=modelOption,temperature=0.7)
else:  
   llm = ChatAnthropic(anthropic_api_key=st.secrets.anthropic_api_key,model=modelOption,temperature=0.7)


#Fetch website

result = st.button("Start")
if result:

  loader = AsyncHtmlLoader(website)
  docs = loader.load()
  html2text = Html2TextTransformer()
  docs_transformed = html2text.transform_documents(docs)
  chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Ein Nutzer stellt dir extrahierte Texte einer Webseite eines Unterhehmens zur Verfügung. Deine Aufgabe ist es folgende Informationen herauszufinden und in deutscher Sprache anzuführen:"
        +"#Name des Unternehmens:# (Wähle dabei den Namen den das Unternehmen in seiner Kommunikation verwendet)\n" 
        +"#Liste angebotener Dienstleistungen und Produkte:# (Wähle dabei maximal 5 Überkategorien\n"
        +"#Vorteile und USPS:# (Wähle dabei die relevantesten USPs. Maximal 4)\n"+
        "#Tonalität und Stil:# (Kommuniziert das Unternehmen per DU mit Kunden, wie förmlich oder locker ist der Stil?"),
        ("human", "{websiteText}")
    ]
)

  messages = chat_template.format_messages(
  websiteText=docs_transformed)
  tsStart = time.time()

  with get_openai_callback() as cb1:
      companyData = llm.invoke(messages).content
      LOGGER.warning("Company Context done after " + str(time.time()-tsStart) + "seconds")
  v_occasion = "<Wichtige Aktion oder Anlass>" + "\n##" + inputOccasion +"##</Wichtige Aktion oder Anlass>" +"\n"
  v_goal ="<Ziel der Nachricht>" + "\n##" + inputGoal + "##</Ziel der Nachricht>" + "\n" 

  chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Du bist ein Kommunikationsspezialist und schreibst 3 Vorschläge einer maßgeschneiderte Push Nachricht auf Deutsch und auf Basis der vom Nutzer bereitgestellten Unternehmensdetails. Beachte die Details um eine möglichst effektive, wahrheitsgemäße Nachricht zugeschnitten auf das Unternehmen zu verfassen und beziehe unbedingt wenn verfügbar spezielle Aktionen oder Anlässe im Title jeder deiner Vorschläge ein. Dabei sollen nur die wichtigsten Top 3 der Dienstleistungen, Produkte, Vorteile und USPs aus den Unternehmensdetails einbezogen werden. Jede Push Nachricht soll aus einem title mit maximal 50 Zeichen und einer description mit maximal 120 Zeichen bestehen. Vermeide Angebote, Vorteile oder Aktionen vorzuschlagen die nicht aus den angegebenen Unternehmensdetails klar und deutlich hervorgehen. \n\nAusgabeformat:\nBegründe jeden deiner Vorschläge mit einem reasoning mit maximal 150 zeichen. Die Ausgabe muss in Deutsch im Markdown Format mit einem JSON Objekt 'pushmessages' mit den 3 Vorschlägen als Objekten. Jedes Objekt enthält eine ID (1-3) title, description und reasoning."),
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
  tsStart = time.time()

  with get_openai_callback() as cb2:
      proposedMessages = llm.invoke(messages)
      LOGGER.warning("Messages done after " + str(time.time()-tsStart) + "seconds")
  st.markdown(proposedMessages.content)
  with st.expander("Verwendeter Company Context"):
    st.markdown(companyData)
  with st.expander("Token Count für Company"):
    st.markdown(cb1)
  with st.expander("Token Count für Messages"):
    st.markdown(cb2)
  
  v_messagesToValidate = proposedMessages.content

  llmValidator = ChatAnthropic(model="claude-3-haiku-20240307", anthropic_api_key=st.secrets.anthropic_api_key, temperature=0)

  chat_template = ChatPromptTemplate.from_messages(
  [
      ("system", "Du erhältst vom Nutzer bis zu 3 Vorschläge für eine Push Benachrichtigung für ein Unternehmen in Form eines JSON Objektes. Bewerte jede dieser Pushbenachrichtungen auf Basis folgender Kriterien:\n <Kriterien>Beachte das keine Inhalte enthalten sind, die so nicht aus den bereitgestellten Unternehmensdetails oder Aktionen, Anlass oder Ziel der Nachricht abgeleitet werden können. Wenn kein Anlass, Aktion oder Ziel in den Unternehmensdetails verfügbar ist, müssen diese auch nicht in den Pushnachrichten bewertet werden. Wenn spezielle Aktionen oder ein Anlass enthalten sind müssen diese in den Nachrichten vorkommen.\nBeachte aber besonders das keine Promotions, Rabatte oder Aktionen enthalten sind die so nicht aus den Unternehmensdetails und Aktionen abgeleitet werden können.\n Beachte ebenso, dass die Nachrichten das Zielpublikum möglichst breit ansprechen sollen.</Kriterien>\n\n<UNTERNEHMENSDETAILS>{companyDetails}\n{occasion}\n{goal}</UNTERNEHMENSDETAILS>\n\nBewertungsskala:\nBewerte in Schulnoten, wobei Note 1 für hervorragende Erfüllung aller Kriterien steht zum Beispiel angegebene Aktionen oder Anlässe im title einbezogen werden. Note 3 wenn wenig Relevanz zu den angegebenen Unternehmensdaten und genannte Aktionen oder Anlässe nicht im titel genannt werden und Note 5 für eine Missachtung der Kriterien, wie zb. 'Aktionen oder Rabatte werden erfunden und sind nicht aus den angegebenen Unternehmensdetails ableitbar' oder kein relevanter Inhalt\n\n Ausgabeformat: Füge die Bewertung mit einem Attribut 'rating' und deine Begründung mit einem Attribut 'critique' jedem Push Benachrichtigungsobjektes hinzu"),
      ("human", "Hier sind die Push Benachrichtigungen in einem JSON Objekt: \n{messagesToValidate}")
  ]
)

  messages = chat_template.format_messages(
      companyDetails=companyData,
      occasion=v_occasion,
      goal=v_goal,
      messagesToValidate=v_messagesToValidate)
  LOGGER.warning(messages)

  validation = llmValidator.invoke(messages)
  st.markdown(validation.content)
  