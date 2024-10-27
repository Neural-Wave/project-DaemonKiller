import streamlit as st
import gc
import uuid
import logging
from rag import RAG
from generation import generate
rag = RAG(n_retrieved=10)
        

# Configure logging for debugging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Streamlit session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.messages = []
    st.session_state.context = None

# Reset chat function
def reset_chat():
    st.session_state.messages = []
    gc.collect()

# Display the application header
col1, col2 = st.columns([6, 1])
with col1:
    st.header("Swisscom AG")
with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         message_placeholder.markdown("Hello, I'm Sam, Swisscom's virtual assisstant. ")

# Capture user input
if prompt := st.chat_input("Ask me anything about Swisscom AG!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    context = rag.query(prompt)
    response = generate(query=prompt, context=context)
    # response_json = response.json()
    

    # # Display user message
    # with st.chat_message("user"):
    #     st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
