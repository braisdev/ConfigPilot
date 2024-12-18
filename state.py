from langgraph.graph import MessagesState
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from typing import Literal


class Classifier(BaseModel):
    """
    This is a classifier that determines if the user input is related to the JSON fields provided.
    """
    related_with_fields: Literal[True, False] = Field(
        ...,
        description="Indicates whether the user input is related to the provided JSON fields."
    )


class RelevanceReflector(BaseModel):
    """
    A utility class to assess and validate the consistency of user input.
    """
    input_type: Literal["ambiguous", "misleading", "stable"] = Field(
        ...,
        description="Nature of the user input: ambiguous, misleading, or stable."
    )
    reasoning: str = Field(
        ...,
        description="Explanation of the input type and its validity."
    )


class AnimeCharacter(BaseModel):
    """
    Represents an anime character with various attributes.
    """
    name: Optional[str] = Field(None, description="Name of the character")
    age: Optional[int] = Field(None, description="Age of the character")
    gender: Optional[str] = Field(None, description="Gender of the character")
    physical_appearance: Optional[str] = Field(None, description="Physical appearance of the character")
    personality: Optional[str] = Field(None, description="Personality of the character")
    abilities_power: Optional[str] = Field(None, description="Abilities or power of the character")
    occupation: Optional[str] = Field(None, description="Occupation of the character")


class ReflectionFeedback(BaseModel):
    """
    Reflection feedback after validating the extracted fields against the user input.
    """
    correctness_summary: str = Field(
        ...,
        description="A brief summary of how accurate and consistent the extracted fields are."
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Serious inconsistencies or errors. Prevents proceeding if non-empty."
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Minor suggestions or improvements that don't block finalization."
    )
    confirmations: List[str] = Field(
        default_factory=list,
        description="Fields that are confirmed correct as-is."
    )
    suggested_corrections: Dict[str, Any] = Field(
        default_factory=dict,
        description="Corrections for fields confidently fixable. If non-empty, proceed_to_creator=false."
    )
    ask_user_about: List[str] = Field(
        default_factory=list,
        description="Fields or issues too contradictory or unclear to resolve. Requires user clarification."
    )
    proceed_to_creator: bool = Field(
        default=True,
        description="Indicates whether to proceed to creator based on feedback."
    )


class GraphState(MessagesState):
    """
    Represents the state of the graph, incorporating various components for comprehensive state management.

    Attributes:
        anime_character: Contains the attributes of the anime character.
        classifier: Determines the relevance of user input to the JSON fields.
        relevance_reflector: Assesses the nature and validity of the user input.
        reflection_feedback: Provides feedback on the extracted fields.
    """
    anime_character: AnimeCharacter
    classifier: Classifier
    relevance_reflector: RelevanceReflector
    reflection_feedback: ReflectionFeedback
    persisted: bool


if __name__ == "__main__":
    # Example instantiation
    graph_state = GraphState(
        classifier=Classifier(related_with_fields=True),
        relevance_reflector=RelevanceReflector(
            input_type="stable",
            reasoning="The user input clearly describes the character's attributes without ambiguity."
        ),
        anime_character=AnimeCharacter(
            name="Sakura",
            age=16,
            gender="Female",
            physical_appearance="Long pink hair, green eyes",
            personality="Cheerful and kind",
            abilities_power="Healing powers",
            occupation="Student"
        ),
        reflection_feedback=ReflectionFeedback(
            correctness_summary="All fields are accurately extracted and consistent with user input.",
            flags=[],
            notes=["Consider adding more details to abilities_power for depth."],
            confirmations=["name", "age", "gender"],
            suggested_corrections={},
            ask_user_about=[],
            proceed_to_creator=True
        )
    )

    # Accessing attributes
    print(graph_state["classifier"].related_with_fields)  # Output: Sakura
    print(graph_state["reflection_feedback"].correctness_summary)
