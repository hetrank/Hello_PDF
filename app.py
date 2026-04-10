# =========================
# FILE: frontend/app.py
# =========================
import streamlit as st
import requests
import json
import time

API_URL = "http://127.0.0.1:8000"

# ----------------------
# Session State
# ----------------------
params = st.query_params
if "token" in params and not st.session_state:
    st.session_state.token = params["token"]
elif "token" not in st.session_state:
    st.session_state.token = None

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------
# Helper
# ----------------------
def get_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

# ----------------------
# Auth UI
# ----------------------
def login_signup():
    st.title("🔐 Login / Signup")

    option = st.radio("Choose", ["Login", "Signup"])

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button(option):
        endpoint = "/login" if option == "Login" else "/signup"

        res = requests.post(
            API_URL + endpoint,
            json={"email": email, "password": password}
        )

        if res.status_code == 200:
            if option == "Login":
                st.session_state.token = res.json()["token"]
                st.query_params["token"] = st.session_state.token
                st.success("Logged in!")
                st.rerun()
            else:
                st.success("Signup successful. Please login.")
        else:
            st.error(res.text)

# ----------------------
# Sidebar (Chats)
# ----------------------
def sidebar():
    st.sidebar.title("💬 Chats")

    # Create new chat
    if st.sidebar.button("➕ New Chat"):
        res = requests.post(
            API_URL + "/chat/create",
            headers=get_headers(),
            params={"name": "New Chat"}
        )

        if res.status_code == 200:
            chat_id = res.json()["id"]
            st.session_state.chat_id = chat_id
            st.session_state.messages = []

    # Fetch chats
    res = requests.get(
        API_URL + "/chat/list",
        headers=get_headers()
    )

    if res.status_code == 200:
        chats = res.json()

        if not chats:
            st.sidebar.info("No chats yet")
            return

        # Create a mapping
        chat_map = {chat["name"] + f" ({chat['id']})": chat["id"] for chat in chats}

        selected_chat = st.sidebar.selectbox(
            "Select Chat",
            options=list(chat_map.keys())
        )

        if selected_chat is None:
            return

        selected_chat_id = chat_map[selected_chat]

        # If changed → load messages
        if st.session_state.get("chat_id") != selected_chat_id:
            st.session_state.chat_id = selected_chat_id

            history_res = requests.get(
                API_URL + f"/chat/{selected_chat_id}/messages",
                headers=get_headers()
            )

            if history_res.status_code == 200:
                st.session_state.messages = history_res.json()   

# ----------------------
# Upload PDFs
# ----------------------
def upload_section():
    st.subheader("📄 Upload PDFs")

    files = st.file_uploader("Upload", accept_multiple_files=True)

    if st.button("Process PDFs") and files:
        for file in files:
            res = requests.post(
                API_URL + f"/chat/{st.session_state.chat_id}/upload",
                headers=get_headers(),
                files={"file": file}
            )

        st.success("Processed!")

# ----------------------
# Chat UI
# ----------------------
def chat_ui():
    st.title("📚 PDF Chat Assistant")

    if not st.session_state.chat_id:
        st.info("Create or select a chat")
        return

    upload_section()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    question = st.chat_input("Ask something...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = requests.post(
                    API_URL + f"/chat/{st.session_state.chat_id}/query",
                    headers=get_headers(),
                    params={"question": question},
                    stream=True
                )

                if res.status_code == 200:
                    full_response = ""
                    sources = []
                    buffer = ""
                    collecting_sources = False
                    sources_buffer = ""

                    placeholder = st.empty()

                    for chunk in res.iter_lines(decode_unicode=True):
                        if chunk:
                            text = chunk.decode("utf-8")

                            if not collecting_sources:
                                buffer += text

                                if "__SOURCES__" in buffer:
                                    parts = buffer.split("__SOURCES__")

                                    full_response += parts[0]
                                    placeholder.markdown(full_response + "▌")

                                    collecting_sources = True
                                    sources_buffer += parts[1]

                                    time.sleep(0.01)
                                
                                else:
                                    full_response += text
                                    placeholder.markdown(full_response)
                            else:
                                sources_buffer += text

                    if sources_buffer:
                        try:
                            sources = json.loads(sources_buffer)
                        except Exception as e:
                            print("JSON parse error: ", e)
                            sources = []

                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                st.write(s)
                    
                    #save to backend
                    requests.post(
                        API_URL+f"/chat/{st.session_state.chat_id}/message",
                        headers=get_headers(),
                        json={
                            "question" : question,
                            "answer" : full_response
                        }
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                else:
                    st.error(res.text)

# ----------------------
# Main
# ----------------------
if not st.session_state.token:
    login_signup()
else:
    sidebar()
    chat_ui()
