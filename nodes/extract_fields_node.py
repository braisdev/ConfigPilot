from chains.field_mapper import map_fields
from state import GraphState


def extract_fields(state: GraphState):
    user_input = state["messages"][-1]

    mapped_fields = map_fields(user_input)
    new_fields = mapped_fields.model_dump(exclude_none=True)

    # Get any previously extracted character fields
    existing_character = state.get("anime_character", {})

    # Merge the old and new fields, where new fields overwrite old ones if they conflict
    merged_character = {**existing_character, **new_fields}

    return {"anime_character": merged_character}
