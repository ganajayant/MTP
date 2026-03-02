import os
import re
from typing import List

import ir_datasets
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import SecretStr
from typing_extensions import TypedDict


class BM25Okapi:
    def __init__(self, corpus):
        pass

    def get_scores(self, query):
        pass


load_dotenv()
g_key = os.getenv("GROQ_API_KEY")

if not g_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")


print("Loading datasets...")

touche = ir_datasets.load("beir/webis-touche2020")
msmarco = ir_datasets.load("msmarco-passage/train")

touche_queries = list(touche.queries_iter())
msmarco_queries = [q.text for q in msmarco.queries_iter()]

touche_docs = {doc.doc_id: doc for doc in touche.docs_iter()}

stance_lookup = {}
for doc in touche.docs_iter():
    if hasattr(doc, "stance"):
        stance_lookup[doc.doc_id] = doc.stance
    else:
        stance_lookup[doc.doc_id] = "UNKNOWN"

tokenized_msmarco = [q.split() for q in msmarco_queries]
bm25_msmarco = BM25Okapi(tokenized_msmarco)

doc_texts = [doc.text for doc in touche_docs.values()]
doc_ids = list(touche_docs.keys())

tokenized_docs = [doc.split() for doc in doc_texts]
bm25_touche = BM25Okapi(tokenized_docs)


class PipelineState(TypedDict):
    query: str
    similar_query: str
    example_docs: List[str]
    test_docs: List[str]
    balanced_example_order: List[int]
    final_ranking: str


def parse_ranking(ranking_str: str):
    numbers = re.findall(r"\[(\d+)\]", ranking_str)
    return [int(n) for n in numbers]


def find_similar_query(state: PipelineState):
    query = state["query"]
    scores = bm25_msmarco.get_scores(query.split())
    best_idx = scores.argmax()
    similar_query = msmarco_queries[best_idx]

    print("Similar Query:", similar_query)

    return {"similar_query": similar_query}


def retrieve_documents(state: PipelineState):
    similar_query = state["similar_query"]
    test_query = state["query"]
    scores_example = bm25_touche.get_scores(similar_query.split())
    top_example_idx = scores_example.argsort()[-10:][::-1]
    example_docs = [doc_ids[i] for i in top_example_idx]
    scores_test = bm25_touche.get_scores(test_query.split())
    top_test_idx = scores_test.argsort()[-10:][::-1]
    test_docs = [doc_ids[i] for i in top_test_idx]

    return {"example_docs": example_docs, "test_docs": test_docs}


def construct_balanced_example(state: PipelineState):
    example_docs = state["example_docs"]

    pro_docs = [d for d in example_docs if stance_lookup.get(d) == "PRO"]
    con_docs = [d for d in example_docs if stance_lookup.get(d) == "CON"]

    balanced = []
    i = j = 0

    while i < len(pro_docs) and j < len(con_docs):
        balanced.append(pro_docs[i])
        balanced.append(con_docs[j])
        i += 1
        j += 1
    balanced.extend(pro_docs[i:])
    balanced.extend(con_docs[j:])
    ranking_indices = [example_docs.index(d) + 1 for d in balanced]

    return {"balanced_example_order": ranking_indices}


llm = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0, groq_api_key=SecretStr(g_key)
)


def llm_rerank(state: PipelineState):
    example_docs = state["example_docs"]
    test_docs = state["test_docs"]
    balanced_order = state["balanced_example_order"]
    similar_query = state["similar_query"]
    query = state["query"]

    example_text = ""
    for i, doc_id in enumerate(example_docs):
        doc = touche_docs[doc_id]
        example_text += f"[{i + 1}] {doc.text[:300]}\n\n"

    example_output = " > ".join([f"[{i}]" for i in balanced_order])

    test_text = ""
    for i, doc_id in enumerate(test_docs):
        doc = touche_docs[doc_id]
        test_text += f"[{i + 1}] {doc.text[:300]}\n\n"

    prompt = f"""
You are RankGPT.

Example:
Query: {similar_query}

{example_text}

Output: {example_output}

Now rank the following documents for this query:

Query: {query}

{test_text}

Return output like:
[1] > [2] > [3]
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"final_ranking": response.content}


graph = StateGraph(PipelineState)  # ty:ignore[invalid-argument-type]

graph.add_node("find_similar_query", find_similar_query)
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("construct_balanced_example", construct_balanced_example)
graph.add_node("llm_rerank", llm_rerank)

graph.set_entry_point("find_similar_query")

graph.add_edge("find_similar_query", "retrieve_documents")
graph.add_edge("retrieve_documents", "construct_balanced_example")
graph.add_edge("construct_balanced_example", "llm_rerank")
graph.add_edge("llm_rerank", END)

app = graph.compile()

if __name__ == "__main__":
    test_query = touche_queries[0].text
    result = app.invoke({"query": test_query})

    print("\nFinal Ranking Output:\n")
    print(result["final_ranking"])

    # Parse ranking order
    ranking_order = parse_ranking(result["final_ranking"])

    # Get ranked document IDs
    ranked_doc_ids = [result["test_docs"][i - 1] for i in ranking_order]

    # Write to output.txt
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(f"Query: {test_query}\n\n")
        f.write("Final Ranking:\n\n")

        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            doc = touche_docs[doc_id]
            f.write(f"Rank {rank}:\n")
            f.write(f"Doc ID: {doc_id}\n")
            f.write(doc.text)
            f.write("\n" + "=" * 80 + "\n\n")

    print("\nRanked documents written to output.txt")
