from chains.relevance_reflector import relevance_reflector
from state import GraphState


def relevance_reflector_node(state: GraphState):

    user_input = state["messages"][-1]

    result = relevance_reflector(user_input)

    return {
        "relevance_reflector": {
            "input_type": result.input_type,
            "reasoning": result.reasoning
        }
    }


