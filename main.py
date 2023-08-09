# import sys 
# print(sys.path)
from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationChain
from langchain.chains import  RetrievalQA
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain.prompts import (
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     ChatPromptTemplate,
#     MessagesPlaceholder
# )
import streamlit as st
from streamlit_chat import message
from utils import *



st.title("Medical Chatbot")
st.subheader("I will make sure you get your answers :)")


if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# if 'buffer_memory' not in st.session_state:

    # st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=3,return_messages=True)
    # st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3)


# system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context,
# and if the answer is not contained within the text below, say 'I don't know'""")

# human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])



# conversation = ConversationChain(memory = st.session_state.buffer_memory, prompt = prompt_template, llm=chat, verbose=True)

conversation = RetrievalQA.from_llm(llm = chat, retriever=retriever)

response_container = st.container()
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing......"):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query")
            st.write(refined_query)
            # context = find_match(refined_query)
            # response = conversation.predict(input=query)
            response = conversation({"query" : refined_query})
        st.session_state.requests.append(query)
        st.session_state.responses.append(response['result'])

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

