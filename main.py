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


@tool
def rank_fairly(query: str):
    """FA*IR fair ranking — ensures demographic groups (gender, race, religion, age,
    disability, nationality, sexuality) are proportionally represented in results.
    Use when the query is about people, representation, or equity across groups."""
    pass


@tool
def rank_diversely(query: str):
    """MMR diverse ranking — maximises variety of viewpoints and topics in results.
    Use when the query is a debate, opinion, or controversial topic that benefits
    from multiple perspectives (e.g. 'should X be banned', 'pros and cons of Y')."""
    pass


@tool
def rank_lexically(query: str):
    """BM25 lexical retrieval — always the first step for every query."""
    pass


@tool
def reranker(query: str):
    """MonoT5 neural reranker — improves relevance for complex, multi-concept, or
    semantic queries where keyword matching alone is insufficient."""
    pass


def get_prompt() -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a document ranking agent. Always call tools in this fixed order:

                STEP 1 (ALWAYS): rank_lexically
                STEP 2 (optional): reranker
                STEP 3 (optional, mutually exclusive): rank_fairly  OR  rank_diversely

                ─── WHEN TO USE EACH OPTIONAL TOOL ───────────────────────────────────────────

                reranker → use when the query is complex, multi-concept, or needs semantic
                           understanding beyond keyword matching.
                           Examples: "effects of climate change on polar bear migration",
                                     "what causes inflation in developing economies"

                rank_fairly → use when the query is about PEOPLE and demographic fairness matters:
                              the results should represent different groups equitably.
                              Trigger keywords / patterns:
                              • mentions of gender, women, men, race, ethnicity, nationality,
                                religion, age, disability, LGBTQ+, minority, diversity
                              • queries about editors, authors, scientists, politicians, athletes,
                                leaders, contributors — where group representation is relevant
                              • "who are the top / best / most influential [people]…"
                              Examples: "female scientists contributions to medicine",
                                        "Wikipedia editors by country",
                                        "notable Black politicians",
                                        "women in technology"

                rank_diversely → use when the query is a DEBATE, OPINION, or CONTROVERSIAL TOPIC
                                 where multiple viewpoints should be surfaced.
                                 Trigger keywords / patterns:
                                 • "should", "pros and cons", "arguments for/against",
                                   "is it good/bad", "debate", "controversial", "opinion"
                                 • policy questions, ethical dilemmas, societal issues
                                 Examples: "should social media be regulated",
                                           "is nuclear energy safe",
                                           "arguments for and against capital punishment"

                ─── DECISION RULES ───────────────────────────────────────────────────────────
                • Use rank_fairly when the query involves people and equitable group representation.
                • Use rank_diversely when the query is a debate/opinion needing varied perspectives.
                • If BOTH could apply, prefer rank_fairly for people-centric queries,
                  rank_diversely for topic/debate-centric queries.
                • Use rank_fairly or rank_diversely even without reranker if the query is simple
                  but clearly has a fairness or diversity dimension.
                • Use the minimum tools needed.

                ─── RESPONSE FORMAT ──────────────────────────────────────────────────────────
                After all tool calls, write ONE sentence explaining why you chose this pipeline.""",
            ),
            ("human", "{input}"),
        ]
    )
    return prompt


def run_with_tools(chain, llm_with_tools, tools_by_name, query):
    """
    Single-pass execution: collect all tool calls from the initial response,
    execute them once, then get final reasoning in one follow-up call.
    Deduplication guard prevents any tool from running twice.
    """
    response = chain.invoke({"input": query})
    messages = [HumanMessage(query), response]

    tools_used = []
    seen_tools = set()

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in seen_tools:
                print(f"  [Skipped duplicate]: {tool_name}")
                continue
            seen_tools.add(tool_name)
            tools_used.append(tool_name)
            result = tools_by_name[tool_name].invoke(tool_call["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

        # Single follow-up to get the reasoning sentence
        final_response = llm_with_tools.invoke(messages)
        reasoning = final_response.content

        if final_response.tool_calls:
            print("[Warning] Extra tool calls after pipeline — ignored.")
    else:
        reasoning = response.content

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_safe = model_name.split("/")[-1].replace(".", "-")
    run_dir = f"{timestamp}_{model_safe}"

    results_dir = os.path.join("results", run_dir)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Results will be saved to: {os.path.abspath(results_dir)}/")

    config_path = os.path.join(results_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"Saved config to: {config_path}")

    if use_local:
        # llm = init_chat_model(
        #     model="openai/gpt-oss-20b",
        #     model_provider="openai",
        #     temperature=0,
        #     openai_api_base="http://127.0.0.1:8000/v1",
        #     openai_api_key="EMPTY",
        # )
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

    tool_kit: List[BaseTool] = [rank_fairly, rank_diversely, rank_lexically, reranker]
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
