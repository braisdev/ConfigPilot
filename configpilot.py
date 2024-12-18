from langgraph.graph import END
from langgraph.graph import StateGraph

from chains.classifier import classify_input
from nodes.creator_node import creator_assistant_node
from nodes.extract_fields_node import extract_fields
from nodes.ambiguity_resolution_node import ambiguity_resolution
from nodes.persist_character import persist_character_fields
from nodes.reflecting_mapping_node import reflect_mapping
from nodes.relevance_reflector_node import relevance_reflector_node
from nodes.non_field_guidance_node import non_field_guidance
from nodes.set_character_fields import set_character_fields

from state import GraphState


def message_related_to_field(state: GraphState):

    print("-- MESSAGE RELATED TO FIELD CONDITION --")

    is_related_to_fields = classify_input(state["messages"][-1]).related_with_fields

    print(f"is related to fields: {is_related_to_fields}")

    if is_related_to_fields:
        print("is related to fields")
        return "relevance_reflector_node"
    else:
        print("is NOT related to fields")
        return "non_field_guidance"


def relevance_reflector_decision(state: GraphState):

    input_type = state["relevance_reflector"]["input_type"]

    if input_type == "stable":
        return "extract_fields_node"
    else:
        return "ambiguity_resolution_node"


def suggested_corrections_decision(state: GraphState):

    if not state["reflection_feedback"]["suggested_corrections"]:
        return "set_character_fields_node"
    else:
        return "extract_fields_node"


builder = StateGraph(GraphState)
builder.add_node("non_field_guidance", non_field_guidance)
builder.add_node("relevance_reflector_node", relevance_reflector_node)
builder.add_node("ambiguity_resolution_node", ambiguity_resolution)
builder.add_node("extract_fields_node", extract_fields)
builder.add_node("reflection_mapping_node", reflect_mapping)
builder.add_node("set_character_fields_node", set_character_fields)
builder.add_node("creator_assistant_node", creator_assistant_node)
builder.add_node("persist_character", persist_character_fields)

builder.set_conditional_entry_point(
    message_related_to_field,
    path_map={
        "non_field_guidance": "non_field_guidance",
        "relevance_reflector_node": "relevance_reflector_node"
    }
)
builder.add_conditional_edges(
    "relevance_reflector_node",
    relevance_reflector_decision,
    path_map={
        "ambiguity_resolution_node": "ambiguity_resolution_node",
        "extract_fields_node": "extract_fields_node"
    }
)

builder.add_conditional_edges(
    "reflection_mapping_node",
    suggested_corrections_decision,
    path_map={
        "set_character_fields_node": "set_character_fields_node",
        "extract_fields_node": "extract_fields_node"
    }
)

builder.add_edge("extract_fields_node", "reflection_mapping_node")
builder.add_edge("set_character_fields_node", "creator_assistant_node")
builder.add_edge("creator_assistant_node", "persist_character")
builder.add_edge("persist_character", END)
graph = builder.compile()
