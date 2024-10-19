from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
from typing import Optional, List, Callable
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool


from .._src.agents.eda_agent import build_eda_agent
from .._src.agents.linear_regression_agent import build_linear_regression_agent

from .._src.llms.openai import build_openai
from .._src.agents.system_prompts.wizard_system_prompt import (
    WIZARD_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
)

from .._src._debug.logger import print_debug


class InitializeEvent(Event):
    pass


class WizardEvent(Event):
    request: Optional[str]
    just_completed: Optional[str]
    need_help: Optional[bool]


class OrchestratorEvent(Event):
    request: str


class EDAEvent(Event):
    request: str


class LinearRegressionEvent(Event):
    request: str


def print_to_user(text: str) -> None:
    """Prints text to the user."""
    print("Wizard: ", str(text))


class WizardWorkflow(Workflow):

    @step(pass_context=True)
    async def initialize(self, ctx: Context, ev: InitializeEvent) -> WizardEvent:

        ctx.data["llm"] = build_openai(temperature=0.3)

        ctx.data["_base_eda_agent"] = build_eda_agent()
        ctx.data["_base_linear_regression_agent"] = build_linear_regression_agent()

        return WizardEvent()

    @step(pass_context=True)
    async def wizard(
        self, ctx: Context, ev: WizardEvent | StartEvent
    ) -> InitializeEvent | WizardEvent | StopEvent:

        print_debug("Wizard received request:", ev.request)

        if "wizard" not in ctx.data:

            agent_worker = FunctionCallingAgentWorker.from_tools(
                tools=[],
                llm=ctx.data["llm"],
                allow_parallel_tool_calls=False,
                system_prompt=WIZARD_SYSTEM_PROMPT,
            )

            ctx.data["wizard"] = agent_worker.as_agent()

        wizard = ctx.data["wizard"]
        if ctx.data["overall_request"] is not None:

            last_request = ctx.data["overall_request"]
            ctx.data["overall_request"] = None
            return OrchestratorEvent(request=last_request)

        elif ev.just_completed is not None:

            response = wizard.chat(
                "The following task has been completed: " + ev.just_completed
            )

        elif ev.need_help:

            print_debug("The previous process needs help with ", ev.request)
            return OrchestratorEvent(request=ev.request)

        else:

            response = wizard.chat("Hello!")

        print_to_user(str(response))

        user_request = input("User request: ").strip()
        return OrchestratorEvent(request=user_request)

    @step(pass_context=True)
    async def orchestrator(
        self, ctx: Context, ev: OrchestratorEvent
    ) -> WizardEvent | EDAEvent | LinearRegressionEvent:

        print_debug("Orchestrator received request:", ev.request)

        def emit_eda() -> bool:
            """Call this if the user wants to perform exploratory data analysis."""
            print_debug("Emitting EDA event.")
            self.send_event(EDAEvent(request=ev.request))
            return True

        def emit_linear_regression() -> bool:
            """Call this if the user wants to perform linear regression."""
            print_debug("Emitting linear regression event.")
            self.send_event(LinearRegressionEvent(request=ev.request))
            return True

        tools = [
            FunctionTool.from_defaults(fn=emit_eda),
            FunctionTool.from_defaults(fn=emit_linear_regression),
        ]

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tools,
            llm=ctx.data["llm"],
            allow_parallel_tool_calls=False,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        )

        ctx.data["orchestrator"] = agent_worker.as_agent()

        orchestrator = ctx.data["orchestrator"]
        response = orchestrator.chat(ev.request)

        if response == "FAILED":
            print_to_user("Orchestrator: FAILED. Try again.")
            return OrchestratorEvent(request=ev.request)

    @step(pass_context=True)
    async def eda(self, ctx: Context, ev: EDAEvent) -> WizardEvent:

        print_debug("EDA received request:", ev.request)

        if "liaison_eda_agent" not in ctx.data:

            def consult_eda_agent(query: str) -> str:
                """Consults the EDA agent with a query."""
                return str(ctx.data["_base_eda_agent"].chat(query))

            system_prompt = """
            You are a helpful assistant acting as a liaison between the user and the EDA agent.
            Simply pass the user's request to the EDA agent.
            Once the EDA agent has completed its task, call the "done" tool.
            If the user asks for something that the EDA agent does not know how to do, call the "need_help" tool.
            """

            ctx.data["liaison_eda_agent"] = LiaisonAgent(
                name="EDA Agent Liaison",
                parent=self,
                fns=[consult_eda_agent],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=EDAEvent,
            )

        return ctx.data["liaison_eda_agent"].handle_event(ev)

    @step(pass_context=True)
    async def linear_regression(
        self, ctx: Context, ev: LinearRegressionEvent
    ) -> WizardEvent:

        print_debug("Linear regression received request:", ev.request)

        if "liaison_linear_regression_agent" not in ctx.data:

            def consult_linear_regression_agent(query: str) -> str:
                """Consults the linear regression agent with a query."""
                return str(ctx.data["_base_linear_regression_agent"].chat(query))

            system_prompt = """
            You are a helpful assistant acting as a liaison between the user and the linear regression agent.
            Simply pass the user's request to the linear regression agent.
            Once the linear regression agent has completed its task, call the "done" tool.
            If the user asks for something that the linear regression agent does not know how to do, call the "need_help" tool.
            """

            ctx.data["liaison_linear_regression_agent"] = LiaisonAgent(
                name="Linear Regression Agent Liaison",
                parent=self,
                fns=[consult_linear_regression_agent],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=LinearRegressionEvent,
            )


class LiaisonAgent:

    def __init__(
        self,
        parent: Workflow,
        fns: List[Callable],
        trigger_event: Event,
        context: Context,
        system_prompt: str,
        name: str,
    ):
        self.parent = parent
        self.fns = fns
        self.trigger_event = trigger_event
        self.context = context
        self.name = name
        self.system_prompt = system_prompt

        def done() -> None:
            """When you complete your task, call this tool."""
            print_debug(f"{self.name} has completed its task.")
            self.context.data["redirecting"] = True

        def need_help() -> None:
            """If the user asks to do something you don't know how to do, call this."""
            print(f"{self.name} needs help")
            self.context.data["redirecting"] = True
            parent.send_event(
                WizardEvent(request=self.current_event.request, need_help=True)
            )

        self.tools = [FunctionTool.from_defaults(fn=fn) for fn in self.fns] + [
            FunctionTool.from_defaults(fn=done),
            FunctionTool.from_defaults(fn=need_help),
        ]

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=self.tools,
            llm=self.context.data["llm"],
            allow_parallel_tool_calls=False,
            system_prompt=system_prompt,
        )

        self.agent = agent_worker.as_agent()

    def handle_event(self, ev: Event):

        self.current_event = ev

        response = str(self.agent.chat(ev.request))
        print_to_user(response)

        if self.context.data["redirecting"]:
            self.context.data["redirecting"] = False
            return None

        user_msg = input("User request: ").strip()
        return self.trigger_event(request=user_msg)


async def main():
    workflow = WizardWorkflow(timeout=1200, verbose=True)
    result = await workflow.run()
    print(result)
