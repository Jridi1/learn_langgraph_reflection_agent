# LinkedIn Post Reflection Agent

> **Learning project.** I built this to learn stateful workflows with LangGraph. It is not production code.

## What it does

An agent that writes a LinkedIn post, critiques it, and rewrites it, in a loop:

1. `generate` writes (or rewrites) the post
2. `reflect` critiques it using a checklist (hook, clarity, structure, tone, value, engagement)
3. The loop stops after 3 revisions

## What I learned

1. How to define state with `TypedDict`, `Annotated` and the `add_messages` reducer
2. Why `StateGraph` is preferred over `MessageGraph`
3. How to build a generate and reflect loop with a conditional edge
4. How to track a `revisions` counter in the state to control when the loop stops
5. How to write a structured critique prompt

## Tech

Python, LangGraph, LangChain, Groq (`openai/gpt-oss-120b`)

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add your own GROQ_API_KEY
python main.py
```

## Graph

![graph](graph.png)


1. Use the critique score as the stopping condition
2. Add tests with a fake LLM
3. Add LangSmith tracing