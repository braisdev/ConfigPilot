from chains.creator_assistant import creation_message
from state import GraphState
import json


def creator_assistant_node(state: GraphState):
    # `confirmations` is a list of field names that were just updated
    new_fields_keys = state["reflection_feedback"]["confirmations"]
    current_fields = state["anime_character"]

    # Extract the newly updated fields from current_fields
    new_fields_dict = {key: current_fields[key] for key in new_fields_keys if key in current_fields}

    # Convert both new and current fields to nicely formatted strings (JSON) for the prompt
    new_fields_str = json.dumps(new_fields_dict, ensure_ascii=False, indent=2)
    current_fields_str = json.dumps(current_fields, ensure_ascii=False, indent=2)

    response = creation_message(new_fields_str, current_fields_str)

    return {"messages": response}
