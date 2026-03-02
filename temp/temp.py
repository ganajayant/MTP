import math
import os

import ir_datasets
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr
from rank_bm25 import BM25Okapi
from typing_extensions import TypedDict

load_dotenv()
g_key = os.getenv("GROQ_API_KEY")

if g_key is None:
    pass


touche = ir_datasets.load("touche/2020/v2")
msmarco = ir_datasets.load("msmarco-passage/train")

touche_queries = list(touche.queries_iter())
msmarco_queries = [q.text for q in msmarco.queries_iter()]
touche_docs = {doc.doc_id: doc for doc in touche.docs_iter()}

stance_lookup = {
    doc.doc_id: getattr(doc, "stance", "UNKNOWN") for doc in touche.docs_iter()
}

# =========================================================
# BM25
# =========================================================

bm25_msmarco = BM25Okapi([q.split() for q in msmarco_queries])

doc_ids = list(touche_docs.keys())
doc_texts = [touche_docs[d].text for d in doc_ids]
bm25_touche = BM25Okapi([t.split() for t in doc_texts])

# =========================================================
# KL GREEDY BALANCING
# =========================================================


def kl_divergence(p, q):
    return sum(p[i] * math.log((p[i] + 1e-9) / (q[i] + 1e-9)) for i in range(len(p)))


def greedy_kl_balance(example_docs, target=(0.5, 0.5)):
    pro = [d for d in example_docs if stance_lookup.get(d) == "PRO"]
    con = [d for d in example_docs if stance_lookup.get(d) == "CON"]

    indices = {"PRO": 0, "CON": 0}
    balanced = []
    counts = {"PRO": 0, "CON": 0}

    total = len(example_docs)

    for _ in range(total):
        candidates = []
        for group in ["PRO", "CON"]:
            if indices[group] < len(pro if group == "PRO" else con):
                candidate = (pro if group == "PRO" else con)[indices[group]]
                candidates.append((group, candidate))

        best_choice = None
        best_score = float("inf")

        for group, doc in candidates:
            temp_counts = counts.copy()
            temp_counts[group] += 1
            current_dist = [
                temp_counts["PRO"] / (len(balanced) + 1),
                temp_counts["CON"] / (len(balanced) + 1),
            ]
            score = kl_divergence(target, current_dist)
            if score < best_score:
                best_score = score
                best_choice = (group, doc)

        if best_choice:
            group, doc = best_choice
            balanced.append(doc)
            counts[group] += 1
            indices[group] += 1

    return [example_docs.index(d) + 1 for d in balanced]


# =========================================================
# AWRF METRIC
# =========================================================


def compute_awrf(ranked_docs):
    exposure = {"PRO": 0.0, "CON": 0.0}
    for i, doc in enumerate(ranked_docs):
        group = stance_lookup.get(doc, "UNKNOWN")
        if group in exposure:
            exposure[group] += 1 / math.log2(i + 2)

    total_exp = sum(exposure.values())
    if total_exp == 0:
        return 0

    dist = [exposure["PRO"] / total_exp, exposure["CON"] / total_exp]
    target = [0.5, 0.5]
    return 1 - kl_divergence(target, dist)


# =========================================================
# SLIDING WINDOW RE-RANK
# =========================================================


def sliding_window_rerank(query, docs, window=10, stride=5):
    reranked = docs.copy()

    for start in range(0, len(docs), stride):
        window_docs = reranked[start : start + window]
        if len(window_docs) < 2:
            continue

        balanced_order = greedy_kl_balance(window_docs)
        new_order = [window_docs[i - 1] for i in balanced_order]

        reranked[start : start + window] = new_order

    return reranked


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=SecretStr(g_key),
)


def run_full_experiment():
    awrf_scores = []

    for query_obj in touche_queries:
        query = query_obj.text

        # Similar query
        scores = bm25_msmarco.get_scores(query.split())
        similar_query = msmarco_queries[np.argmax(scores)]

        # Retrieve docs
        scores_test = bm25_touche.get_scores(query.split())
        top_idx = scores_test.argsort()[-20:][::-1]
        test_docs = [doc_ids[i] for i in top_idx]

        # Sliding KL-based re-rank
        final_docs = sliding_window_rerank(query, test_docs)

        # Compute fairness
        awrf = compute_awrf(final_docs[:10])
        awrf_scores.append(awrf)

    print("Average AWRF:", np.mean(awrf_scores))


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    run_full_experiment()
