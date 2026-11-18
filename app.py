import streamlit as st
import chromadb
from openai import OpenAI
from gtts import gTTS
from dotenv import load_dotenv
import os

# 1. Setup
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

col = chromadb.PersistentClient(
    path="./cars_db"
).get_collection("car_data")

# 2. Streamlit Interface
st.title("AI Car Salesman")
st.write("Ask me anything about cars!")

query = st.text_input("What kind of car are you looking for?")

def chat_with_cars(query):
    # Retrieve
    context = "\n".join(
        col.query(query_texts=[query], n_results=3)["documents"][0]
    )

    # Prompt for readable and speakable output
    prompt = f"Car Salesman. Use ONLY context. Be concise. Format: ## Heading, **Bold**, Bullets.\n\nContext: {context}\n\nUser: {query}"

    # Generate
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[{"role": "user", "content": prompt}]
    )

    text_result = response.choices[0].message.content

    # Speech Synthesis
    clean_text = text_result.replace("#", "").replace("*", "")
    tts = gTTS(clean_text)
    tts.save("response.mp3")

    # Display results
    st.markdown(text_result)
    st.audio("response.mp3", autoplay=True)

# 3. Run
if st.button("Ask"):
    if query:
        chat_with_cars(query)