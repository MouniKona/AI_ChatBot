import streamlit as st
from RAG import initialize_milvus, query_rag

with st.spinner("Initializing Milvus"):
    vector_store = initialize_milvus()

# Initialize the session state for storing chat messages and chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # Current chat messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of saved chats, each chat is a list of messages
if "active_chat_index" not in st.session_state:
    st.session_state.active_chat_index = None  # Tracks which chat is currently active

# Chat header
st.title("Enhanced Chatbot Interface")
st.write("This is a simple chatbot interface with saved chat history on the left.")

# Sidebar for saved chat history
with st.sidebar:
    st.header("Chat History")

    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            chat_name = f"Chat {i + 1} ({len(chat)} messages)"
            if st.button(chat_name, key=f"chat_{i}"):
                st.session_state.messages = chat.copy()  # Load selected chat history
                st.session_state.active_chat_index = i  # Set the active chat index
                st.rerun()  # <-- CHANGED FROM st.experimental_rerun() TO st.rerun()

    if st.button("New Chat"):
        if st.session_state.messages:  # Save current chat if it has messages
            if st.session_state.active_chat_index is not None:
                # Update the existing chat in history if the user was in a previous chat
                st.session_state.chat_history[st.session_state.active_chat_index] = st.session_state.messages.copy()
            else:
                # Save the new chat if it was not linked to an existing history
                st.session_state.chat_history.append(st.session_state.messages.copy())
        st.session_state.messages = []  # Clear chat messages for new chat
        st.session_state.active_chat_index = None  # Reset active chat index
        st.rerun()  # <-- CHANGED FROM st.experimental_rerun() TO st.rerun()

with st.spinner("Initializing Milvus"):
    vector_store = initialize_milvus()

# Display existing chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# Input field for user input
if prompt := st.chat_input("Ask me anything!"):
    # Save the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # If the user is in an active chat, update the corresponding chat in history
    if st.session_state.active_chat_index is not None:
        st.session_state.chat_history[st.session_state.active_chat_index] = st.session_state.messages.copy()

    # Display user's message
    with st.chat_message("user"):
        st.write(prompt)

    # Simulated assistant response
    # response = f"You said: '{prompt}' - I'm a simulated assistant!"
    response = query_rag(prompt)
    # Save assistant's response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # If the user is in an active chat, update the corresponding chat in history
    if st.session_state.active_chat_index is not None:
        st.session_state.chat_history[st.session_state.active_chat_index] = st.session_state.messages.copy()

    # Display assistant's response
    with st.chat_message("assistant"):
        st.write(response)
