from langchain_core.messages import HumanMessage

from configpilot import graph


if __name__ == "__main__":

    messages = [HumanMessage(content="el personaje de anime se llama Brais tiene 14 años")]
    print(graph.invoke({"messages": messages}))
