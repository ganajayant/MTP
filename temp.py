import csv
import os
from datetime import datetime
from typing import Dict, List

import ir_datasets
import tqdm
import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr


def get_dotenv() -> Dict[str, SecretStr]:
    load_dotenv()
    g_key = os.getenv("GROQ_API_KEY")
    if not g_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return {"GROQ_API_KEY": SecretStr(g_key)}


def load_config(path="config/run.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_datasets(datasets: list):
    print("Loading datasets...")
    for dataset in datasets:
        print(f"Loading dataset: {dataset}")
        ds = ir_datasets.load(dataset)
        queries = list(ds.queries_iter())
        docs = list(ds.docs_iter())
        print(f"Loaded {len(queries)} queries and {len(docs)} docs for: {dataset}")


# ── Pipeline 1 ────────────────────────────────────────────────────────────────
@tool
def pipeline_lexical(query: str):
    """Pipeline: rank_lexically
    BM25 lexical retrieval only.
    Use for simple, short, keyword queries where the terms in the query
    directly match what you would expect in relevant documents.
    Examples: "python tutorial", "french revolution causes", "capital of Brazil"
    """
    pass


# ── Pipeline 2 ────────────────────────────────────────────────────────────────
@tool
def pipeline_lexical_reranker(query: str):
    """Pipeline: rank_lexically → reranker
    BM25 retrieval followed by MonoT5 neural reranking.
    Use for complex, multi-concept, or semantic queries where keyword
    matching alone is insufficient and deeper relevance scoring is needed.
    Examples: "effects of climate change on polar bear migration",
              "what causes inflation in developing economies",
              "long-term impact of antibiotics on gut microbiome"
    """
    pass


# ── Pipeline 3 ────────────────────────────────────────────────────────────────
@tool
def pipeline_lexical_reranker_fairly(query: str):
    """Pipeline: rank_lexically → reranker → rank_fairly
    BM25 retrieval, neural reranking, then FA*IR fair ranking.
    Use when the query is about PEOPLE and equitable demographic representation
    matters (gender, race, ethnicity, nationality, religion, age, disability,
    sexuality). Ensures results proportionally represent different groups.
    Trigger patterns: mentions of gender · women · men · race · ethnicity ·
    nationality · religion · age · disability · LGBTQ+ · minority · diversity;
    queries about editors, authors, scientists, politicians, athletes, leaders,
    contributors where group representation is relevant; "who are the top /
    best / most influential [people]…"
    Examples: "female scientists contributions to medicine",
              "Wikipedia editors by country",
              "notable Black politicians",
              "women in technology"
    """
    pass


# ── Pipeline 4 ────────────────────────────────────────────────────────────────
@tool
def pipeline_lexical_reranker_diversely(query: str):
    """Pipeline: rank_lexically → reranker → rank_diversely
    BM25 retrieval, neural reranking, then MMR diverse ranking.
    Use when the query is a DEBATE, OPINION, or CONTROVERSIAL TOPIC where
    surfacing multiple viewpoints and perspectives is important.
    Trigger patterns: "should" · "pros and cons" · "arguments for/against" ·
    "is it good/bad" · "debate" · "controversial" · "opinion"; policy
    questions, ethical dilemmas, societal issues.
    Examples: "should social media be regulated",
              "is nuclear energy safe",
              "arguments for and against capital punishment",
              "pros and cons of remote work"
    """
    pass


def get_prompt() -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a document ranking agent. For every query, call EXACTLY ONE of
the following four pipeline tools:

  1. pipeline_lexical
     → Simple keyword queries; terms map directly to expected document content.

  2. pipeline_lexical_reranker
     → Complex, multi-concept, or semantic queries needing deeper relevance scoring.

  3. pipeline_lexical_reranker_fairly
     → Queries about PEOPLE where equitable demographic representation matters
       (gender, race, nationality, religion, age, disability, sexuality, etc.).

  4. pipeline_lexical_reranker_diversely
     → Debate, opinion, or controversial queries that benefit from multiple
       viewpoints (use "should", "pros and cons", "arguments for/against", etc.).

─── DECISION RULES ────────────────────────────────────────────────────────────
• If the query is simple and keyword-based                → pipeline_lexical
• If the query is complex or semantic                     → pipeline_lexical_reranker
• If the query involves people + fairness/representation  → pipeline_lexical_reranker_fairly
• If the query is a debate / opinion / controversial      → pipeline_lexical_reranker_diversely
• When fairness AND diversity both apply, prefer
  pipeline_lexical_reranker_fairly for people-centric queries,
  pipeline_lexical_reranker_diversely for topic/debate-centric queries.

─── RESPONSE FORMAT ───────────────────────────────────────────────────────────
Call exactly one pipeline tool, then write ONE sentence explaining your choice.""",
            ),
            ("human", "{input}"),
        ]
    )
    return prompt


def run_with_tools(chain, llm_with_tools, tools_by_name, query):
    """
    Single-pass execution: call exactly one pipeline tool from the initial
    response, execute it, then get the final reasoning sentence.
    """
    response = chain.invoke({"input": query})
    messages = [HumanMessage(query), response]

    tools_used = []

    if response.tool_calls:
        # Only the first tool call is honoured (should always be exactly one)
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tools_used.append(tool_name)

        if len(response.tool_calls) > 1:
            print(
                f"  [Warning] LLM emitted {len(response.tool_calls)} tool calls; "
                "only the first will be used."
            )

        result = tools_by_name[tool_name].invoke(tool_call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        # Single follow-up to get the reasoning sentence
        final_response = llm_with_tools.invoke(messages)
        reasoning = final_response.content

        if final_response.tool_calls:
            print("[Warning] Extra tool calls in follow-up response — ignored.")
    else:
        reasoning = response.content
        print(f"  [Warning] No tool called for: {query}")

    return reasoning, tools_used


def process_dataset(dataset_name, chain, model_with_tools, tools_by_name, results_dir):
    print(f"\n{'=' * 60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'=' * 60}")

    dataset = ir_datasets.load(dataset_name)
    queries = list(dataset.queries_iter())

    results = []
    for query in tqdm.tqdm(queries, desc=dataset_name, total=len(queries)):
        reasoning, tools_used = run_with_tools(
            chain, model_with_tools, tools_by_name, query.text
        )
        if not tools_used:
            print(f"  [Warning] No tools used for: {query.text}")

        results.append(
            {
                "query": query.text,
                "tools_used": "|".join(tools_used),
                "reasoning": reasoning,
            }
        )

    safe_name = dataset_name.replace("/", "_")
    output_path = os.path.join(results_dir, f"{safe_name}_result.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "tools_used", "reasoning"])
        writer.writeheader()
        writer.writerows(results)

    print(f"  -> Saved: {output_path}")
    return results


def main():
    keys = get_dotenv()
    config = load_config()
    datasets = config["datasets"]

    model_name = config["model"]

    temperature = config["sampling"]["temperature"]
    top_p = config["sampling"]["top_p"]
    presence_penalty = config["sampling"]["presence_penalty"]

    use_local = config["use_local"]
    should_load_datasets = config["load_dataset"]

    if should_load_datasets:
        load_datasets(datasets)

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe: str = model_name.split("/")[-1].replace(".", "-")
    run_dir: str = f"{timestamp}_{model_safe}"

    results_dir: str = os.path.join("results", run_dir)
    os.makedirs(name=results_dir, exist_ok=True)

    print(f"Results will be saved to: {os.path.abspath(results_dir)}/")

    config_path: str = os.path.join(results_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"Saved config to: {config_path}")

    if use_local:
        llm = init_chat_model(
            model=model_name,
            model_provider="openai",
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            openai_api_base="http://127.0.0.1:8000/v1",
            openai_api_key="EMPTY",
        )
    else:
        llm = init_chat_model(
            model_name,
            model_provider="groq",
            temperature=0,
            groq_api_key=keys["GROQ_API_KEY"],
        )

    print(f"Initialized LLM: {model_name}")

    tool_kit: List[BaseTool] = [
        pipeline_lexical,
        pipeline_lexical_reranker,
        pipeline_lexical_reranker_fairly,
        pipeline_lexical_reranker_diversely,
    ]
    tools_by_name = {t.name: t for t in tool_kit}

    prompt = get_prompt()
    model_with_tools = llm.bind_tools(tool_kit)
    chain = prompt | model_with_tools

    for dataset_name in datasets:
        process_dataset(
            dataset_name, chain, model_with_tools, tools_by_name, results_dir
        )

    print(f"\nDone. Results in: {os.path.abspath(results_dir)}/")


if __name__ == "__main__":
    main()
