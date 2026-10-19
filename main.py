from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

generation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional LinkedIn content assistant tasked with crafting engaging, "
     "insightful and well structured LinkedIn posts. "
     "Generate the best LinkedIn post possible for the user's request. "
     "If the user provides feedback or critique, respond with a refined version of your "
     "previous attempt, improving clarity, tone and engagement as needed."),
    MessagesPlaceholder(variable_name="messages"),
])
generate_chain = generation_prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional LinkedIn content strategist and critic.
You review the latest LinkedIn post in the conversation and give feedback so it can be improved in the next revision.

Evaluate the post using these guidelines:
1. Hook: Does the first line make people stop scrolling? Only about 2 lines show before "see more".
2. Clarity: Is there one clear main message that is easy to follow?
3. Structure: Short paragraphs, good spacing, logical flow, no wall of text. Ideal length is around 120 to 250 words.
4. Tone: Professional but human, no buzzwords or cliches, sounds like a real person.
5. Value: Does the reader learn something, get an insight, or take away something useful?
6. Engagement: Is there a clear call to action or a question that invites comments? Hashtags (3 to 5) and emojis should be relevant and used sparingly.

Provide a detailed critique that includes:
* Strengths: what works well and should be kept
* Weaknesses: specific problems, quoting the exact lines concerned
* Suggestions: a concrete fix for each weakness
* Score: a rating out of 10 with one sentence of justification

Your critique will be used to improve the post in the next revision step.
Be specific, constructive, and actionable. Do not rewrite the whole post yourself."""),
    MessagesPlaceholder(variable_name="messages"),
])
reflect_chain = reflection_prompt | llm


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    revisions: int


def generation_node(state: AgentState) -> dict:
    post = generate_chain.invoke({"messages": state["messages"]})
    return {
        "messages": [post],
        "revisions": state.get("revisions", 0) + 1,
    }


def reflection_node(state: AgentState) -> dict:
    critique = reflect_chain.invoke({"messages": state["messages"]})
    # Sent as a HumanMessage so the generator treats it as feedback
    return {"messages": [HumanMessage(content=critique.content)]}


def should_continue(state: AgentState) -> str:
    if state["revisions"] >= 3:
        return END
    return "reflect"


workflow = StateGraph(AgentState)
workflow.add_node("generate", generation_node)
workflow.add_node("reflect", reflection_node)
workflow.set_entry_point("generate")
workflow.add_conditional_edges("generate", should_continue, {END: END, "reflect": "reflect"})
workflow.add_edge("reflect", "generate")
app = workflow.compile()


if __name__ == "__main__":
    initial_state = {
        "messages": [HumanMessage(content="I want to teach others the difference between langchain and langgraph")],
        "revisions": 0,
    }
    result = app.invoke(initial_state)
    for m in result["messages"]:
        m.pretty_print()