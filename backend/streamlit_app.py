import streamlit as st
import asyncio
from radchat import Pipe # Assuming radchat and Pipe are correctly set up

st.title("RadChat Interface")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Pipe - Placeholder for your actual Pipe initialization
# Ensure this is done only once or as needed.
if "pipe" not in st.session_state:
    st.session_state.pipe = Pipe() # Or however your Pipe is initialized

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # If the content was originally a dict, it would have been stored as a formatted string
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare message for pipe
    # Consider how much history your pipe needs.
    # Sending all messages: st.session_state.messages
    # Sending only the last user message: [{"role": "user", "content": prompt}]
    pipe_input = {
        "messages": st.session_state.messages # Sending all messages for context
    }

    # Get response from pipe
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Run the async pipe function
                response_content = asyncio.run(st.session_state.pipe.pipe(pipe_input)) # Use pipe from session state

                # parts = []
                # print(citations)
                # for citation in citations:
                #     for key, value in citation.items():
                #         parts.append(f"- **{key.replace('_', ' ')}:** {value}")
                    
                #     # For storage, you might want a consistent string representation
                #     # Here, we'll store a simple representation.
                #     # If you want to re-render dicts specifically from history, you'll need a more complex message structure.
                # response_content += "\nCitations:\n" "\n".join(parts)


                st.markdown(response_content)
                assistant_response_content = response_content

                st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})