from chains.ambiguity_resolution import generate_response
from state import GraphState


def ambiguity_resolution(state: GraphState):

    user_input = state["messages"][-1]
    reason = state["relevance_reflector"]["reasoning"]

    response = [generate_response(user_input, reason)]

    messages = state["messages"] + response

    return {"messages": messages}

