from state import GraphState


def set_character_fields(state: GraphState):

    anime_character = state.get("anime_character", {})

    return {
        "anime_character": {
            "name": anime_character.get("name"),
            "age": anime_character.get("age"),
            "gender": anime_character.get("gender"),
            "physical_appearance": anime_character.get("physical_appearance"),
            "personality": anime_character.get("personality"),
            "abilities_power": anime_character.get("abilities_power"),
            "occupation": anime_character.get("occupation"),
        }
    }
