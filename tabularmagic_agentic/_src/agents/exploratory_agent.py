from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from ..tools.general_tools import (
    data_overview_tool,
)
from ..tools.exploratory_tools import (
    test_equal_means_tool,
)
from ..shared_analyzer import shared_analyzer
from ..env_vars import env_vars
import tabularmagic as tm


def create_exploratory_agent(
    analyzer: tm.Analyzer,
    llm: ChatOpenAI | None = None,
    verbose: bool = False,
) -> AgentExecutor:
    """Create an exploratory agent."""

    shared_analyzer.set_shared_analyzer(analyzer)

    if llm is None:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", api_key=env_vars.get_openai_key(), temperature=0
        )
    tools = [data_overview_tool, test_equal_means_tool]
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        ),
    )

    return AgentExecutor(agent=agent, tools=tools, verbose=verbose)
