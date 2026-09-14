import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="College Assistant", page_icon="🎓", layout="centered")

# ----------------------------------------------------------------------------
# Backend setup (cached so it only runs once, not on every Streamlit rerun)
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Setting up the assistant (loading PDFs, building indexes)...")
def load_workflow():
    api = os.getenv("OPENROUTER_API_KEY")
    api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

    if not api:
        raise ValueError("OPENROUTER_API_KEY is missing from .env")

    model = ChatOpenAI(
        model="openrouter/free",
        api_key=api,
        base_url="https://openrouter.ai/api/v1",
    )

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=api_key
    )

    def build_retriever(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        document = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(document)

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        return vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

    acedmic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")

    class State(TypedDict):
        program: str
        messages: Annotated[list, add_messages]
        query_type: str
        retrieved_context: str

    def classifier_node(state: State) -> dict:
        last_message = state["messages"][-1].content

        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"

            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotion, course structure, summer training, or degree requirements.\n"

            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"

            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"

            f"Query: {last_message}\n\n"

            "Return only one word: academic, fee, or general."
        )

        response = model.invoke(prompt)

        category = response.content.strip().lower()

        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"

        return {"query_type": category}

    def academic_rag_node(state: State) -> dict:
        query = state["messages"][-1].content

        docs = acedmic_retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        return {"retrieved_context": context}

    def fee_rag_node(state: State) -> dict:
        query = state["messages"][-1].content

        docs = fee_retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        return {"retrieved_context": context}

    def general_node(state: State) -> dict:
        return {
            "retrieved_context": "NO_RETRIEVAL_NEEDED"
        }

    def response_node(state: State) -> dict:
        query = state["messages"][-1].content
        program = state.get("program", "unknown")
        context = state["retrieved_context"]

        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant talking to a {program} student. "
                f"Answer this question using your own general knowledge:\n\n"
                f"{query}"
            )
        else:
            prompt = (
                f"You are a college assistant helping a {program} student. "
                f"Use the following context from the official college documents to answer "
                f"the question accurately. If the context mentions specific figures for "
                f"different programs, highlight the one relevant to {program} if possible.\n\n"

                f"Context:\n{context}\n\n"

                f"Question: {query}\n\n"

                f"Give a clear, friendly, and precise answer."
            )

        response = model.invoke(prompt)

        return {
            "messages": [
                ("ai", response.content.strip())
            ]
        }

    def route_query(state: State):
        if state["query_type"] == "academic":
            return "academic_rag"

        elif state["query_type"] == "fee":
            return "fee_rag"

        else:
            return "general"

    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")

    graph.add_conditional_edges(
        "classifier",
        route_query
    )

    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    workflow = graph.compile()

    return workflow


workflow = load_workflow()

# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------

st.title("🎓 College Assistant")
st.caption("Ask about academics, fees, or anything else — I'll route your question automatically.")

st.sidebar.header("Your Program")

selected_program = st.sidebar.radio(
    "Which college program are you in?",
    ["BCA", "BBA", "B.Com (H)"],
)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.info(f"You're set as a **{selected_program}** student.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message("user" if role == "human" else "assistant"):
        st.markdown(content)

user_query = st.chat_input("Type your question...")

if user_query:
    st.session_state.chat_history.append(("human", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = workflow.invoke({
                    "program": selected_program,
                    "messages": [
                        ("human", user_query)
                    ]
                })
                answer = result["messages"][-1].content
            except Exception as e:
                answer = f"⚠️ Something went wrong: {e}"

        st.markdown(answer)

    st.session_state.chat_history.append(("ai", answer))




