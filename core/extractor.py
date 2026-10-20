from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
def get_llm():
  return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"), temperature=0.3)


def build_chain(system_prompt :str):
    llm = get_llm()
    prompt=ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", {"text"}),
    ]
    )
    chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text": x}) | prompt | llm | StrOutputParser()
    )
    return chain

def extract_action_items(transcript :str) -> str:
    system_prompt = "You are an expert meeting analyst. From the meeting transcript,"
    "extract all the action items discussed during the meeting. For each action item, provide:\n"
    "-Task Description: A clear and concise description of the task.\n"
    "-Owner(Responsible Person): The individual or team responsible for completing the task.\n"
    "-Due Date: The deadline or expected completion date for the task.(if mentioned else write 'Not Specified')\n"
    "-Priority Level: The urgency or importance of the task (e.g., High, Medium, Low). If not mentioned, write 'Not Specified'.\n"
    "Format the output as a numbered list of action items, with each item containing the above details. If no action items are found, respond with 'No action items were discussed in the meeting"

    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})

def extract_key_decisions(transcript :str) -> str:
    system_prompt = "You are an expert meeting analyst. From the meeting transcript, extract all the key decisions made during the meeting. For each decision, provide:\n"
    "-Decision Description: A clear and concise description of the decision.\n"
    "-Owner(Responsible Person): The individual or team responsible for implementing the decision.\n"
    "-Date: The date when the decision was made.(if mentioned else write 'Not Specified')\n"
    "Format the output as a numbered list of key decisions, with each decision containing the above details. If no key decisions are found, respond with 'No key decisions were made in the meeting'"
    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})

def extract_questions(transcript :str) -> str:
    system_prompt = "You are an expert meeting analyst. From the meeting transcript,"
    "extract all the questions asked during the meeting. For each question, provide:\n"
    "-Question Description: A clear and concise description of the question.\n"
    "-Owner(Responsible Person): The individual or team responsible for providing the answer.\n"
    "-Due Date: The deadline or expected completion date for the question.(if mentioned else write 'Not Specified')\n"
    "Format the output as a numbered list of questions, with each question containing the above details. If no questions are found, respond with 'No questions were asked in the meeting'"

    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})


