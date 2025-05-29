"""
Core chatbot logic + memory-driven SOP flow
Import `physio_reply()` in FastAPI (or any backend) to get an answer string.
"""

import os, re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()                                               
embeddings = OpenAIEmbeddings()
vec_store  = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever  = vec_store.as_retriever(search_kwargs={"k": 12})
llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


GREETING_PATTERNS = [
    r'\b(?:hi|hello|hey|greetings|howdy|good morning|good afternoon|good evening)\b',
    r'^(?:hi|hello|hey|yo)$', r'^(?:hi|hello|hey|yo)\s+.*$'
]

SOP_QUESTIONS = [
    "What problem are you experiencing?",
    "What is the exact location of the pain/problem?",
    "How long has the pain/problem been present?",
    "On a scale of 1-10, how would you rate the intensity of your pain?",
    "Are you experiencing any movement restrictions due to this pain/problem?",
    "Do you have any history of injury at this location?",
    "What event likely caused this issue? (It's okay if you don't know)",
    "do you have any comorbidities like diabetes and hypertension?",
    "What kind of job/profession do you have?"
]

SYS_PROMPT = """
You are PhysioGPT, an AI physiotherapy assistant with expertise in musculoskeletal issues.
• Analyze the patient information carefully:
  - Problem: {problem}
  - Location: {location}
  - Duration: {duration}
  - Pain Intensity: {intensity}/10
  - Movement Restrictions: {restrictions}
  - Injury History: {history}
  - Possible Cause: {cause}
  - Comorbidities : {comorbidities}
  - Profession: {profession}

RESPONSE STRUCTURE:
1. Warm, empathetic acknowledgment (1-2 sentences)
2. Likely causes (2-3 sentences)
3. Two evidence-based home exercises – clear instructions, reps & frequency
4. Realistic recovery expectations
5. Safety disclaimer about red-flag symptoms

IMPORTANT:
• Conversational, encouraging tone
• No medication advice
• Jargon-free language
""".strip()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm        = llm,
    retriever  = retriever,
    return_source_documents=False,
)
qa_chain.combine_docs_chain.llm_chain.prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYS_PROMPT),
    SystemMessagePromptTemplate.from_template("Relevant extracts:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template("{question}")
])


class _SOPState:
    def __init__(self): self.reset()
    def reset(self):
        self.current = 0
        self.answers = dict.fromkeys(
            ["problem","location","duration","intensity",
             "restrictions","history","cause","comorbidities","profession"], "")
        self.complete = False


state = _SOPState()

def _is_greeting(txt:str) -> bool:
    txt = txt.lower().strip()
    return any(re.search(p, txt) for p in GREETING_PATTERNS)

def _process(message:str, history:list[list[str]]) -> str:
    global state
    
    
    if not history: 
        state.reset()

    
    if not state.complete:
        
        if state.current == 0:
            if _is_greeting(message) and len(message.split()) <= 5:
                return ("Hello! I'm PhysioGPT. To help you better, "
                        "please describe the problem you're experiencing.")
            state.answers["problem"] = message
            state.current = 2
            return SOP_QUESTIONS[1]

        
        keys = ["problem","location","duration","intensity",
                "restrictions","history","cause","comorbidities","profession"]
        state.answers[keys[state.current-1]] = message
        state.current += 1

        
        if state.current > len(SOP_QUESTIONS):
            state.complete = True
            summary = "\n".join([f"- {k.replace('_',' ').title()}: {v}"
                                 for k,v in state.answers.items()])
            patient_prompt = f"Patient Information:\n{summary}\n\nBased on this information, provide a complete physiotherapy assessment including likely causes, recommended exercises, recovery expectations, and safety information."
            
            
            chat_hist = []
            for pair in history:
                if len(pair) == 2:  
                    chat_hist.append(HumanMessage(content=pair[0]))
                    chat_hist.append(AIMessage(content=pair[1]))
            
            
            response = qa_chain.invoke({
                "question": patient_prompt,
                "chat_history": chat_hist,
                **state.answers
            })
            return response["answer"]
        else:
            return SOP_QUESTIONS[state.current-1]

    
    
    chat_hist = []
    for pair in history:
        if len(pair) == 2:  
            chat_hist.append(HumanMessage(content=pair[0]))
            chat_hist.append(AIMessage(content=pair[1]))

   
    response = qa_chain.invoke({
        "question": message,
        "chat_history": chat_hist,
        **state.answers
    })
    return response["answer"]


def physio_reply(message:str, history:list[list[str]]) -> str:
    """
    Parameters
    ----------
    message : str
        Latest user text.
    history : list of [user, ai] pairs (strings) accumulated in the browser.

    Returns
    -------
    str : chatbot reply.
    """
    return _process(message, history or [])