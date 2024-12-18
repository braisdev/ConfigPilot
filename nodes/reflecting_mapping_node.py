from chains.reflection_mapper import reflect_on_extraction
from state import GraphState
from state import AnimeCharacter
from pydantic import ValidationError


def reflect_mapping(state: GraphState):

    user_input = state["messages"][-1]

    anime_character_dict = state["anime_character"]

    try:
        anime_character = AnimeCharacter.model_validate(anime_character_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid AnimeCharacter data: {e}")

    reflection = reflect_on_extraction(user_input, anime_character)

    # Return the reflection feedback in the desired format
    return {"reflection_feedback": reflection.model_dump()}
