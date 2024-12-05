# Import para Agente e Ferramentas
from langchain.agents import AgentExecutor, ConversationalAgent, Tool
# Import para Ferramentas
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
import os
from dotenv import load_dotenv
# Import para Memória e Histórico de Mensagens
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import (StreamlitCallbackHandler)
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
# Import para o modelo de linguagem
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
# Streamlit
import streamlit as st
# Google Cloud Credentials
from google.cloud.bigquery.client import Client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'data\core-shard-442902-h7-3fded6207039.json'
bq_client = Client()


# Importar Chaves de API
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
finance_wrapper = GoogleFinanceAPIWrapper(serp_api_key=SERPAPI_API_KEY)

def init_memory():
    """
    Iniciar a memória para o Chatbot
    """
    return ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
MEMORY = init_memory()
CHAT_HISTORY = MessagesPlaceholder(variable_name="chat_history")

# Criar ferramentas que serão utilizadas
tools = [
    Tool(
        name="search",
        func=search.run,
        description="Search the web for information."
        # Pesquisa na Web detalhes sobre a mensagem recebida.
    ),
    GoogleFinanceQueryRun(api_wrapper=finance_wrapper)
    # Pesquisa detalhes nas páginas do Google Finance.
]

# Criar o Prefixo e Sufixo para o Agente
prefix = """
    You are a chatbot created to receive a question about a certain company or stock and provide information about it. You will search information in Google search and in the Google Finance page.
    You must display the name of the company and stock, the current price, the change in price, and as much information as you can find about the stock.
    You can also search the web for news and important details about said company or stock and should give all the information you can find to the user. Always be formal and polite.
    In case you don't know the answer, be clear and explain if the company doesn't exist or if there is no information available. Only answer in at least 2 paragraphs, don't use bullet points or lists.
    In these paragraphs include information about the company always. NEVER give the user a suggestion to buy a stock or invest in a company.
"""

suffix = """
Chat History:
{chat_history}
Latest Question: {input}
{agent_scratchpad}
"""

# Iniciar o Agente
prompt = ConversationalAgent.create_prompt(
    tools,
    prefix = prefix,
    suffix = suffix,
    input_variables = {"input", "chat_history", "agent_scratchpad"}
)

# Cache na memória
msg = StreamlitChatMessageHistory()
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        messages = msg,
        memory_key = "chat_history",
        return_messages=True
    )
memory = st.session_state["memory"]

# Chamar modelo
llm_chain = LLMChain(
    llm = ChatGoogleGenerativeAI(temperature=0.5, model="gemini-1.5-flash", api_key=GEMINI_API_KEY),
    prompt=prompt,
    verbose=True
)

# Agente e Executor
agent = ConversationalAgent(
    llm_chain=llm_chain,
    memory=memory,
    max_interactions=10,
    tools = tools
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, memory=memory, tools=tools, handle_parsing_errors=True)

# Customização Streamlit
st.set_page_config(page_title="Finance Chatbot", page_icon="🤖")

st.header("Welcome to the Google Finance Chatbot!")
st.markdown("#### This chatbot is designed to provide information about companies and stocks. You can ask questions about a company or stock and the chatbot will search for information on Google and Google Finance to provide you with the most accurate information available.")

# Botão para excluir o histórico de mensagens
with st.sidebar:
    st.write("Use this button to clear the chat history.")
if st.sidebar.button("Clear message history"):
    MEMORY.chat_memory.clear()

# Chatbot
avatars = {
    "human": "user",
    "ai": "assistant"
}
for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)
    
if prompt := st.chat_input(placeholder="Which stock or company would you like to know more about?"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt} ,
            {"callbacks": [st_callback]}
        )
        st.write(response["output"])

