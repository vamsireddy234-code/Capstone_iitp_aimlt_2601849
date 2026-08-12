from typing import TypedDict
from langgraph.graph import StateGraph, START , END
from sentence_transformers import SentenceTransformer
import chromadb
import os
from groq import Groq
import json
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn

prompt = [{"role" : "You are a helpful assistant who already answers about zepto policies "},
          {"context" : "You are only authorised to only look into available docs {context}"},
          {"task" : "Analyse the question : {query} and answer from {context}"},
          {"format": "Return a JSON object with three fields: answer as a string, sources as a list of source or chunk IDs, and confidence as a float between 0 and 1."},
          {"length": "Keep the answer limited to the information needed to answer the user's question."},
          {"negative" : "Please dont give own created information beyond the document , if you dont find and you say i don't have information"},
          {"confidence": "give out confidance in float between 0 and 1"},
          {"example" : """Example Question : what is the delivery fee for an oder below 149?
                          Example Context : Order below INR 149 incur of INR 25 delivery fee
                            
                          Example Answer : 
                            {
                                "answer": "Orders below INR 149 incur a flat INR 25 delivery fee.",
                                "sources": ["doc_01.txt"],
                                "confidence": 1.0
                             
                            } """}]

groq_client = Groq(
    api_key="dummy_key")
mock_llm = os.getenv("MOCK_LLM", "1")


class ragstate(TypedDict):
    query : str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


class answerschema(BaseModel):
    answer : str
    sources : list[str]
    confidence : float = Field(ge=0.0, le=1.0)

def classify_intent(state):
    query = state["query"].lower()
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

    isfound = False

    for keyword in keywords:
        if keyword in query:
            isfound = True
            break

    if isfound:
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        **state,
        "intent" : intent
            }

folder = os.path.dirname(os.path.abspath(__file__))

chrome_db_path = os.path.join(folder, "chroma_db")

client = chromadb.PersistentClient(path=chrome_db_path)

collection = client.get_collection(name="Zepto_Policies")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_and_answer(state):

    query = state["query"]

    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    ids = results["ids"][0]
    #embeddings = results["embeddings"][0]

    context = "\n\n".join(documents)


    #source = ids

    final_prompt = f"""
        You are a helpful assistant who answers about Zepto policies.

        Use only the following available policy documents:

        Context: {context}

        User question: {query}

        Answer only from the provided context.
        If the answer is not available in the context, say:
        "I don't have information about that in the available documents."

        Return a JSON object with:
            answer: string
            sources: list of relevant source IDs
            confidence: float between 0 and 1
        """

    if mock_llm == "1":
            top_chunk = documents[0]

            top_chunk_snippet = top_chunk[:200]

            result = answerschema(
                answer = f"Based on the retrieved context: {top_chunk_snippet}",

                sources = [ids[0]],

                confidence = 1.0
            )

            answer = result.answer
            sources = result.sources
            confidence = result.confidence

    else:
        response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful Zepto policy assistant."
            },
            {   
                "role": "user",
                "content": final_prompt
                    }
                ]
            )

        answer = response.choices[0].message.content

        final_answer = json.loads(answer)
        answer = final_answer["answer"]
        sources =final_answer["sources"]
        confidence = final_answer["confidence"]

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "confidence" : confidence
    }


def direct_answer(state):
    
    query = state["query"]

    result = answerschema(
            answer = "I don't have information about that in the available documents.",
            sources = [],
            confidence = 0.0
    )

    return {
        **state,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence
    }


graph = StateGraph(ragstate)

graph.add_node("classify_intent" , classify_intent)

graph.add_node("retrieve_and_answer" , retrieve_and_answer)

graph.add_node("direct_answer", direct_answer)


def route_intent(state):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    else:
        return "direct_answer"



graph.add_edge(START, "classify_intent")

graph.add_conditional_edges("classify_intent", route_intent, {

"retrieve_and_answer" : "retrieve_and_answer",

"direct_answer" : "direct_answer"

})

graph.add_edge("retrieve_and_answer", END)

graph.add_edge("direct_answer", END)


app = graph.compile()

test_state = {
    "query": "What is the delivery fee?",
    "intent": "",
    "answer": "",
    "sources": [],
    "confidence": 0.0
}

ans = app.invoke(test_state)

print(ans)

api = FastAPI()

class askrequest(BaseModel):
    query : str


@api.post("/ask" , response_model= answerschema)

def ask(request : askrequest):

    state = {

        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }


    result = app.invoke(state)

    return answerschema(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=8000)


