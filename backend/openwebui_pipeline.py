"""
TODO: 
    - This needs to enable an openwebui openai compatible server or pipeline
    - The pipeline will call another server or will be a server
    - The algorithm will be standalone and not necessarily interfaced through openwebui
    - There needs to be a single streamlit app that can also interface with the llm server
    - The llm server will be like ollama but offer our models that have much more logic than just the llm call
"""


from langchain_google_genai import ChatGoogleGenerativeAI


model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", timeout=600)
import streamlit as st

# read the bom text
with open("../data/bom/bom.txt", "r") as f:
    bom_text = f.read()


# Streamlit app
st.title("Book of Mormon Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response from Gemini
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        prompt = f"Please read the following book then answer the prompt: {bom_text}\n\n\n Here is the user's prompt:" + prompt
        
        
        full_response = model.invoke(prompt).content
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
