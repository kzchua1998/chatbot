import streamlit as st
import requests
import subprocess
import time

st.title("Job Recommendation Machine")

if "messages" not in st.session_state:
    st.session_state["messages"] = list()
    welcome_msg = "Hello there, tell me about your job history, skills, and experience in not less than 10 words for more accurate recommendation!"
    # st.chat_message("assistant").markdown(welcome_msg)
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# React to user input
if prompt := st.chat_input("Tell me about your job responsibilities, skillsets, and years of experience"):

    # Display user input in the chat
    user_prompt = f"{prompt}"
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Make a request to the FastAPI endpoint for predictions
    api_url = "http://127.0.0.1:8001/predict"  # Adjust the URL based on your FastAPI setup
    response = requests.post(api_url, json={"text": prompt})

    if response.status_code == 200:
        predictions = response.json()
        recommended_jobs = predictions["recommended_job"]
        predicted_spec = predictions["predicted_spec"]

        # Display bot response with predictions
        bot_response = f"Based on you description, you are specialized in {predicted_spec}.\n \
            These are the recommended jobs for you\n{recommended_jobs}"
        time.sleep(1)
        st.chat_message("assistant").markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    else:
        st.error(f"Failed to get predictions. Error {response.status_code}: {response.text}")
