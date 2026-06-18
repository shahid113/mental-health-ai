import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import json
import uuid
import logging
import streamlit as st
from typing import Annotated, TypedDict

# LangChain & LangGraph Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Reranking Imports
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Transformers Import
from transformers import pipeline

# ==========================================
# UI Setup & Configuration
# ==========================================
st.set_page_config(page_title="Mental Health AI Companion", page_icon="💙", layout="centered")

st.title("💙 AI Mental Health Companion")
st.warning(
    "**IMPORTANT DISCLAIMER**: This is an educational prototype. It is NOT a substitute for professional mental health care. "
    "If you are in crisis, please contact a professional immediately (e.g., dial 988 in the US, or 112 in India)."
)

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Configuration")
    hf_token = st.text_input("Hugging Face API Token", type="password", help="Get this from your Hugging Face settings.")
    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    else:
        st.info("Please enter your Hugging Face API Token to start.")
        st.stop()
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ==========================================
# State Definition
# ==========================================
class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    intent: str
    sentiment: str
    is_crisis: bool
    retrieved_knowledge: str
    coping_strategies: str
    final_response: str

# ==========================================
# Cached Resource Initialization
# ==========================================
@st.cache_resource(show_spinner="Initializing AI Models & Knowledge Base... This may take a minute on first load.")
def load_agent():
    logging.basicConfig(level=logging.WARNING)
    
    # 1. LLM Init
    hf_endpoint = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        temperature=0.7,
        max_new_tokens=1024,
        return_full_text=False,
        do_sample=True
    )
    llm = ChatHuggingFace(llm=hf_endpoint)

    # 2. Knowledge Base & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    file_path = "mental_health_knowledge.json"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({
                "knowledge_base": [{
                    "topic": "Anxiety",
                    "subtopics": {
                        "coping_techniques": [
                            "Deep breathing exercises: Try 4-7-8 breathing.",
                            "Grounding techniques: 5-4-3-2-1 sensory exercise."
                        ]
                    }
                }]
            }, f)

    with open(file_path, "r") as f:
        raw_data = json.load(f)

    docs = []
    for topic_data in raw_data.get("knowledge_base", []):
        topic_name = topic_data.get("topic", "")
        for subtopic_name, content in topic_data.get("subtopics", {}).items():
            clean_subtopic = subtopic_name.replace("_", " ").title()
            content_str = "\n- " + "\n- ".join(content) if isinstance(content, list) else str(content)
            chunk = f"Topic: {topic_name}\nSubtopic: {clean_subtopic}\nInformation: {content_str}"
            docs.append(chunk)

    vectorstore = Chroma.from_texts(
        texts=docs,
        embedding=embeddings,
        collection_name="mental_health_kb",
        persist_directory="./chroma_health_db"
    )

    # 3. Reranking Retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=2)
    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

    # 4. Emotion Pipeline
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

    # 5. Agent Definitions
    def router_agent(state: ConversationState):
        user_input = state["user_input"]
        context = f"\nPrevious Bot Message: {state['messages'][-2].content}\n" if len(state["messages"]) > 1 else ""
        prompt = ChatPromptTemplate.from_template(
            "Classify the user's input intent into EXACTLY ONE of the following categories:\n"
            "- 'greeting': Simple hellos, how are you.\n"
            "- 'mental_health_query': Questions about anxiety, stress, therapy, or FOLLOW-UPS to the previous message.\n"
            "- 'crisis': Severe distress, self-harm, emergencies.\n"
            "- 'off_topic': Coding, math, trivia, general non-health topics.\n\n"
            "{context}User message: {user_input}\n\nRespond with JUST the category word."
        )
        response = (prompt | llm).invoke({"context": context, "user_input": user_input})
        intent_raw = response.content.strip().lower()
        intent = "mental_health_query"
        for valid in ["greeting", "mental_health_query", "crisis", "off_topic"]:
            if valid in intent_raw: intent = valid; break
        return {"intent": intent}

    def off_topic_handler_agent(state: ConversationState):
        response = "I am a specialized mental health assistant. I'm not equipped to answer general trivia or off-topic questions. However, if you'd like to talk about how you're feeling or manage stress, I'm here for you!"
        return {"final_response": response, "messages": [AIMessage(content=response)]}

    def sentiment_analyzer_agent(state: ConversationState):
        try:
            result = emotion_pipeline(state["user_input"])[0]
            sentiment_map = {"joy": "positive", "surprise": "neutral", "neutral": "neutral", "sadness": "distressed", "fear": "distressed", "anger": "negative", "disgust": "negative"}
            sentiment = sentiment_map.get(result['label'], "neutral")
        except:
            sentiment = "neutral"
        return {"sentiment": sentiment}

    def crisis_detector_agent(state: ConversationState):
        prompt = ChatPromptTemplate.from_template("Detect if this message indicates a mental health crisis requiring immediate help (suicide, self-harm, urgent danger):\nUser message: {user_input}\nRespond with: yes or no")
        response = (prompt | llm).invoke({"user_input": state["user_input"]})
        return {"is_crisis": "yes" in response.content.strip().lower()}

    def knowledge_retrieval_agent(state: ConversationState):
        search_query = state["user_input"]
        if len(state["messages"]) > 1 and len(search_query.split()) < 8:
            search_query = f"Context: {state['messages'][-2].content[:200]}... Query: {search_query}"
        try:
            relevant_docs = retriever.invoke(search_query)
            return {"retrieved_knowledge": "\n\n".join([doc.page_content[:500] for doc in relevant_docs])}
        except:
            return {"retrieved_knowledge": "General mental health support information available."}

    def counselor_agent(state: ConversationState):
        history_text = "\n".join([f"{'👤 User' if m.type == 'human' else '🤖 Counselor'}: {m.content}" for m in state["messages"][:-1][-6:]])
        prompt = ChatPromptTemplate.from_template(
            "You are an empathetic mental health counselor.\n--- Conversation History ---\n{history}\n--------------------------\n\n"
            "Current User: {user_input}\nRelevant Knowledge: {retrieved_knowledge}\n\n"
            "Provide warm, supportive guidance directly answering the current user prompt while using the history for context. Do NOT diagnose."
        )
        response = (prompt | llm).invoke({"history": history_text, "user_input": state["user_input"], "retrieved_knowledge": state.get("retrieved_knowledge", "")})
        return {"final_response": response.content.strip()}

    def coping_strategy_agent(state: ConversationState):
        prompt = ChatPromptTemplate.from_template("Based on this information, suggest 2-3 practical coping strategies:\n{retrieved_knowledge}\nFormat as bullet points. Keep each strategy to one sentence.")
        response = (prompt | llm).invoke({"retrieved_knowledge": state.get("retrieved_knowledge", "")})
        return {"coping_strategies": response.content.strip()}

    def response_formatter_agent(state: ConversationState):
        if state.get("coping_strategies") and state.get("intent") != "greeting":
            formatted_response = f"{state.get('final_response', '')}\n\n**Practical strategies you might try:**\n{state.get('coping_strategies', '')}"
        else:
            formatted_response = state.get("final_response", "")
        return {"final_response": formatted_response, "messages": [AIMessage(content=formatted_response)]}

    def crisis_handler_agent(state: ConversationState):
        crisis_response = (
            "🚨 **CRISIS SUPPORT - IMMEDIATE HELP NEEDED** 🚨\n\n"
            "I'm concerned about what you're experiencing. Please reach out for immediate professional help:\n\n"
            "📞 **Emergency Resources:**\n"
            "• National Suicide Prevention Lifeline: **988** (24/7, free, confidential)\n"
            "• India: iCall **9152987821** or dial **112**\n"
            "• Crisis Text Line: Text **HOME** to **741741**\n"
            "• International: https://findahelpline.com\n\n"
            "You deserve support, and help is available right now. Please reach out immediately."
        )
        return {"final_response": crisis_response, "messages": [AIMessage(content=crisis_response)]}

    # 6. Build Graph
    workflow = StateGraph(ConversationState)
    workflow.add_node("router", router_agent)
    workflow.add_node("off_topic_handler", off_topic_handler_agent)
    workflow.add_node("sentiment", sentiment_analyzer_agent)
    workflow.add_node("crisis_detector", crisis_detector_agent)
    workflow.add_node("crisis_handler", crisis_handler_agent)
    workflow.add_node("retrieve_knowledge", knowledge_retrieval_agent)
    workflow.add_node("counselor", counselor_agent)
    workflow.add_node("coping", coping_strategy_agent)
    workflow.add_node("formatter", response_formatter_agent)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", lambda s: "off_topic_handler" if s.get("intent") == "off_topic" else "sentiment", {"off_topic_handler": "off_topic_handler", "sentiment": "sentiment"})
    workflow.add_edge("off_topic_handler", END)
    workflow.add_edge("sentiment", "crisis_detector")
    workflow.add_conditional_edges("crisis_detector", lambda s: "crisis_handler" if s.get("is_crisis") else "retrieve_knowledge", {"crisis_handler": "crisis_handler", "retrieve_knowledge": "retrieve_knowledge"})
    workflow.add_edge("crisis_handler", END)
    workflow.add_edge("retrieve_knowledge", "counselor")
    workflow.add_edge("counselor", "coping")
    workflow.add_edge("coping", "formatter")
    workflow.add_edge("formatter", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

app = load_agent()

# ==========================================
# Chat Interface & Memory Management
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("How are you feeling today?"):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # LangGraph Config (Binding the session to LangGraph Memory)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {
        "user_input": prompt,
        "messages": [HumanMessage(content=prompt)]
    }

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Reflecting..."):
            result = app.invoke(initial_state, config=config)
            bot_response = result['final_response']
        st.markdown(bot_response)
        
    st.session_state.messages.append({"role": "assistant", "content": bot_response})