import streamlit as st
import requests

# Page Config
st.set_page_config(page_title="AI Research Scout", page_icon="🧬", layout="wide")

st.title("🧬 AI Research Scout")
st.markdown("Enter a topic to find the latest ArXiv research and get a synthesized 'Golden Nugget' idea.")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    thread_id = st.text_input("User Session ID", value="default_user")
    api_url = st.text_input("FastAPI URL", value="http://localhost:8000/research")

# Main Input
topic = st.chat_input("What research topic are you interested in?")

if topic:
    # 1. Display User Message
    with st.chat_message("user"):
        st.write(topic)

    # 2. Call the FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Scouting papers and synthesizing ideas..."):
            try:
                response = requests.post(
                    api_url, 
                    json={"topic": topic, "thread_id": thread_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.subheader("💡 The Golden Nugget Idea")
                    st.info(data["final_idea"])
                    
                    st.caption(f"Processed {data['papers_found']} papers from ArXiv.")
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Could not connect to Backend: {e}")

# Bottom Info
st.divider()
st.markdown("Developed with LangGraph, FastAPI, and Gemini 2.5 Pro.")