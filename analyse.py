import csv
import os
from typing import Dict, List

import ir_datasets
import tqdm
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


def load_datasets(datasets: list):
    print("Loading datasets...")
    for dataset in datasets:
        print(f"Loading dataset: {dataset}")
        ir_datasets.load(dataset)
        print(f"Indexing dataset: {dataset}")
        ds = ir_datasets.load(dataset)
        queries = list(ds.queries_iter())
        docs = list(ds.docs_iter())
        print(
            f"Loaded {len(queries)} queries and {len(docs)} documents for dataset: {dataset}"
        )


@tool
def rank_fairly(query: str):
    """Rank documents fairly using FA*IR algorithm for queries involving protected attributes (gender, race, etc.). Use after rank_lexically."""
    pass


@tool
def rank_diversely(query: str):
    """Rank documents diversely using MMR to maximize result variety. Use after rank_lexically."""
    pass


@tool
def rank_lexically(query: str):
    """Retrieve relevant documents using BM25 lexical matching. Always call this first before other ranking tools."""
    pass


@tool
def reranker(query: str):
    """Rerank documents by relevance using MonoT5. Use after rank_lexically for complex queries."""
    pass


def run_with_tools(chain, llm_with_tools, tools_by_name, query):
    response = chain.invoke({"input": query})
    messages = [HumanMessage(query), response]

    tools_used = []

    while response.tool_calls:
        for tool_call in response.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            tools_used.append(tool_call["name"])
            result = tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

        # Subsequent calls can use llm_with_tools directly (context is in messages)
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    return response.content, tools_used


def main():
    keys = get_dotenv()

    datasets = [
        "msmarco-passage/trec-dl-2019",
        "beir/webis-touche2020",
        "trec-fair/2021/train",
    ]

    OPTIONS = {
        "dataset": "beir/webis-touche2020",
        "model": "openai/gpt-oss-120b",
        "load_datasets": False,
        "use_local": True,
    }

    if OPTIONS["load_datasets"]:
        load_datasets(datasets)

    print(keys, OPTIONS)

    if OPTIONS["use_local"]:
        llm = init_chat_model(
            model="openai/gpt-oss-20b",
            model_provider="openai",
            temperature=0,
            openai_api_base="http://127.0.0.1:8000/v1",
            openai_api_key="EMPTY",
        )
    else:
        llm = init_chat_model(
            OPTIONS["model"],
            model_provider="groq",
            temperature=0,
            groq_api_key=keys["GROQ_API_KEY"],
        )

    print(f"Initialized LLM: {OPTIONS['model']}")

    tool_kit: List[BaseTool] = [rank_fairly, rank_diversely, rank_lexically, reranker]

    # tempprompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             """
    #         You are a Ranking Agent that ranks documents based on the query. You have access to the following tools:
    #             1. rank_fairly: Ranks documents fairly based on the FA*IR algorithm.
    #             2. rank_diversely: Ranks documents diversely based on diversity criteria.
    #             3. rank_lexically: Ranks documents based on BM25 lexical matching.
    #             4. reranker: Reranks documents based on relevance to the query using models such as MonoT5.

    #         For now the methods for the tools are not defined but you can still call the tools and they will return dummy outputs. You should try to use the tools as much as possible to ensure good relevance, fairness and diversity in the results.

    #         Generally, based on the query certain tools can be used let's say if the query is simple and doesn't need any reranking then rank_lexically can be used, if the query is complex and needs relevance based reranking then reranker can be used, if the query has protected attributes and needs fair ranking then rank_fairly can be used, if the query needs diverse results then rank_diversely can be used. You can also use multiple tools for a single query. Always try to use the most appropriate tool for the query and try to use as less tools as possible while ensuring good relevance and fairness. Always provide reasoning for why you chose a particular tool for a query in your response.

    #         Note: if your using tools like rank_fairly or rank_diversely or reranker then you should first use rank_lexically to get relevant documents based on BM25 lexical matching and then use the other tools on the relevant documents to ensure good relevance and fairness.
    #       """,
    #         ),
    #         # MessagesPlaceholder("chat_history", optional=True),
    #         ("human", "{input}"),
    #         # MessagesPlaceholder("agent_scratchpad"),
    #     ]
    # )

    tempprompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a document ranking agent. You must select and call tools in the
                following strict order — never skip a step, never call a tool out of order:
                  STEP 1 — rank_lexically   (MANDATORY for every query, always first)
                  STEP 2 — reranker         (optional: use for complex / semantic queries)
                  STEP 3 — rank_fairly      (optional: use if the query involves protected attributes
                                              such as gender, race, religion, age, disability)
                           OR
                           rank_diversely   (optional: use if the query is broad or benefits from
                                              multiple perspectives / varied viewpoints)
                           (rank_fairly and rank_diversely are mutually exclusive per query)
                Rules:
                  • Always call rank_lexically first.
                  • Only add reranker when BM25 alone is not enough (multi-concept, nuanced queries).
                  • Only add rank_fairly when the query involves equity / demographic representation.
                  • Only add rank_diversely when the query is broad, controversial, or needs variety.
                  • Use the minimum number of tools that satisfies the query's needs.
                  • After calling all required tools, write a concise 1–2 sentence explanation of
                    WHY you chose this specific pipeline for the query.
                    Use the minimum tools necessary. Briefly state your reasoning.""",
            ),
            ("human", "{input}"),
        ]
    )
    model_with_tools = llm.bind_tools(tool_kit)
    chain = tempprompt | model_with_tools

    tools_by_name = {t.name: t for t in tool_kit}
    model_with_tools = llm.bind_tools(tool_kit)

    dataset = ir_datasets.load(OPTIONS["dataset"])
    queries = list(dataset.queries_iter())
    results = []
    for query in tqdm.tqdm(queries, desc="Processing queries", total=len(queries)):
        answer, tools_used = run_with_tools(
            chain, model_with_tools, tools_by_name, query.text
        )
        if len(tools_used) == 0:
            print(f"Warning: No tools used for query: {query.text}")
        results.append({"query": query.text, "tools_used": tools_used})

    # print("Full response object:")
    # print(response)
    # print("\nContent:", response.content)
    # print("\nTool calls:", response.tool_calls)
    # print("\nAdditional kwargs:", response.additional_kwargs)

    safe_dataset_name = OPTIONS["dataset"].replace("/", "_")
    result_name = f"{safe_dataset_name}_result.csv"
    with open(result_name, "w", newline="") as csvfile:
        fieldnames = ["query", "tools_used"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":
    main()
