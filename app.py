import streamlit as st
from ollama import chat

st.set_page_config(
    page_title="Local Qwen Chat",
    layout="wide"
)

st.title("🤖 Local AI Chat (Qwen + Ollama)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
prompt = st.chat_input("Ask anything...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        response = chat(
            model="qwen:latest",
            messages=st.session_state.messages
        )

        output = response.message.content

        placeholder.write(output)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": output
        }
    )