from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
def get_llm():
  return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"), temperature=0.3)


def split_transcript(transcript :str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return text_splitter.split_text(transcript)

def summarize(transcript :str) -> str:
     llm = get_llm()
     map_prompt=ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely, highlighting the key points and main ideas discussed. Ensure that the summary is clear, coherent, and captures the essence of the meeting without including unnecessary details."),
        ("human", "{text}"),
    ]
    )

     map_chain = map_prompt | llm | StrOutputParser()   

     chunks=split_transcript(transcript)
     chunk_summaries=[map_chain.invoke({"text": chunk}) for chunk in chunks]

     combined = "\n\n".join(chunk_summaries)
     combined_prompt=ChatPromptTemplate.from_messages([
        ("system", "You are an expert meeting summarizer. Combine these partial simmaries "
                    "into a single, coherent professional meeting summary that captures the key points and main ideas discussed in bullet points."
                    " Please ensure that the summary is clear, coherent, and captures the essence of the video without including unnecessary details."),
        ("human", "{text}"),
    ]
    )
     combined_chain = (
         RunnablePassthrough() | RunnableLambda(lambda x:{"text": x}) | combined_prompt | llm | StrOutputParser()
     )

     return combined_chain.invoke({combined})

def generate_title(transcript :str) -> str:
    llm = get_llm()
 
    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text": x}) | ChatPromptTemplate.from_messages([
            ("system", "You are an expert meeting summarizer. Generate a concise and informative title for this meeting transcript that captures the essence of the discussion in a few words."
             "(max 8 words ) Only return the title, do not include any additional text or formatting."),
            ("human", "{text}"),
        ]) | llm | StrOutputParser()
    )
    return title_chain.invoke({"text": transcript})