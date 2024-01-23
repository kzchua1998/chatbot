import streamlit as st
import requests
import yaml
import time

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

streamlit_data = config["streamlit"]
APP_TITLE = streamlit_data["app_title"]
WELCOME_MSG = streamlit_data["welcome_msg"]
CHAT_INPUT_MSG = streamlit_data["chat_input_msg"]
WARNING_MSG = streamlit_data["warning_msg"]
API_URL = streamlit_data["api_url"]

st.title(APP_TITLE)

if "messages" not in st.session_state:
    st.session_state["messages"] = list()
    st.session_state.messages.append({"role": "assistant", "content": WELCOME_MSG})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# React to user input
if prompt := st.chat_input(CHAT_INPUT_MSG):

    # Display user input in the chat
    user_prompt = f"{prompt}"
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    if len(prompt.split(" ")) >= 10:

        # Make a request to the FastAPI endpoint for predictions
        response = requests.post(API_URL, json={"text": prompt})

        if response.status_code == 200:
            predictions = response.json()
            recommended_jobs = predictions["recommended_job"]
            predicted_spec = predictions["predicted_spec"]

            time.sleep(1)

            with st.chat_message("assistant"):
                assistant_response_newline = ["Based on you description, these are the recommended jobs for you!  \n"]
                i = 0
                for job_title, job_description in recommended_jobs:
                    assistant_response_newline.append(f"{i+1}) {job_title} - {job_description}  \n")
                    i += 1

                # Simulate stream of responses
                full_response = ""
                message_placeholder = st.empty()
                for chunk in assistant_response_newline:
                    full_response += chunk + " "
                    time.sleep(0.1)
                    message_placeholder.markdown(full_response)

                assistant_response = ''.join(assistant_response_newline)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        else:
            st.error(f"Failed to get predictions. Error {response.status_code}: {response.text}")
    else:
        time.sleep(1)
        st.chat_message("assistant").markdown(WARNING_MSG)
        st.session_state.messages.append({"role": "assistant", "content": WARNING_MSG})