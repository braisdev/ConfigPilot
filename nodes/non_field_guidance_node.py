from typing import Dict, Any

from chains.non_field_guidance import provide_related_info
from state import GraphState


def non_field_guidance(state: GraphState) -> Dict[str, Any]:

    print("---NON FIELD GUIDANCE---")
    last_user_message = state["messages"][-1]

    related_info = [provide_related_info(last_user_message)]
    messages = state["messages"] + related_info

    return {"messages": messages}
