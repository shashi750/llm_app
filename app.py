import streamlit as st
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Hugging Face pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=generator)

# LangChain prompt
prompt = PromptTemplate(
    input_variables=["topic", "tone"],
    template="Write a {tone} paragraph about {topic}."
)
chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit UI
st.title("LLM Text Generator with LangChain + Hugging Face")
topic = st.text_input("Enter a topic", "Artificial Intelligence")
tone = st.selectbox("Select a tone", ["excited", "serious", "funny", "sad", "neutral"])

if st.button("Generate"):
    with st.spinner("Generating..."):
        response = chain.run({"topic": topic, "tone": tone})
        st.success("Result:")
        st.write(response)
