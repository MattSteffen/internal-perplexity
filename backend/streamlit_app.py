import streamlit as st
import asyncio
from radchat import Pipe

st.title("RadChat Interface")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Pipe
pipe = Pipe()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# add query: "what is the louvain algorithm"
default_prompt = "What is the louvain algorithm?"
# Add user message to chat history
st.session_state.messages.append({"role": "user", "content": default_prompt})
with st.chat_message("user"):
    st.markdown(default_prompt)

# Prepare message for pipe
pipe_input = {
    "messages": st.session_state.messages
}

# Get response from pipe
with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        # Run the async pipe function
        response = asyncio.run(pipe.pipe(pipe_input))
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response}) 


# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare message for pipe
    pipe_input = {
        "messages": st.session_state.messages
    }

    # Get response from pipe
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run the async pipe function
            response = asyncio.run(pipe.pipe(pipe_input))
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response}) 