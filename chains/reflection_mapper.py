from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate


class AnimeCharacter(BaseModel):
    """Represents an anime character."""
    name: Optional[str] = Field(None, description="Name of the character")
    age: Optional[int] = Field(None, description="Age of the character")
    gender: Optional[str] = Field(None, description="Gender of the character")
    physical_appearance: Optional[str] = Field(None, description="Physical appearance of the character")
    personality: Optional[str] = Field(None, description="Personality of the character")
    abilities_power: Optional[str] = Field(None, description="Abilities or power of the character")
    occupation: Optional[str] = Field(None, description="Occupation of the character")


from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class ReflectionFeedback(BaseModel):
    """
    Reflection feedback after validating the extracted fields against the user input.

    Fields:
      correctness_summary:
        - A brief summary of how accurate and consistent the extracted fields are.

      flags:
        - Serious inconsistencies or errors that must be resolved before proceeding.
        - Unlike ask_user_about, flags might be used for issues that are clearly incorrect but not necessarily
          requiring user input—just that they must be fixed before continuing.

      notes:
        - Minor suggestions or improvements that do not block finalization.

      confirmations:
        - Fields confirmed correct as-is (including correct None values).

      suggested_corrections:
        - Fields that can be confidently corrected based on the user input.
        - If non-empty, proceed_to_creator must be false.

      ask_user_about:
        - Fields or issues that cannot be confidently resolved from the given input due to contradictions or ambiguity.
        - Indicates the need to request further clarification from the user.
        - If non-empty, proceed_to_creator must be false.

      proceed_to_creator:
        - True if no flags, no suggested_corrections, and no ask_user_about issues are present.
        - False otherwise.
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


def reflect_on_extraction(user_input: str, extracted_character: AnimeCharacter) -> ReflectionFeedback:
    """
    Given the user input and the extracted AnimeCharacter fields, perform a reflection step.
    The reflection checks for internal consistency, whether fields were correctly extracted,
    and provides a summary along with possible corrections.
    """

    system_prompt = SystemMessage(
        """You are a strict fact-checking assistant. You have two inputs:
        1. The original user input describing an anime character.
        2. The extracted fields from a previous step.
        
        Rule: The user input doesn't have to contain all the fields, we're just checking the fields that the user input
        and extracted character provides, ignoring Nones, null, and empty values if not mentioned in the user input.

        Your categories:
        - flags: Internal pipeline issues or severe logical errors that can't be resolved from current data or user clarification. 
                 The system must fix these before proceeding.
        - ask_user_about: Contradictions or ambiguities in the user input that can only be resolved by asking the user.
        Never add here fields that the user do not mention. 
        - suggested_corrections: Discrepancies that can be confidently fixed based on the user input. 
        - notes: Minor suggestions or improvements that do not block progress.
        - confirmations: Fields that are correct as is.

        If a field is not mentioned, it remains None without suggested corrections.
        If a field can be corrected from user input, add to suggested_corrections.
        If user input is contradictory or unclear for a field, add that field to ask_user_about.
        If there is a serious logical/pipeline error, add a flag.
        Non-blocking advice goes in notes.
        Fields correct as-is go in confirmations.

        Output only JSON:
        {
          "correctness_summary": "...",
          "flags": [...],
          "notes": [...],
          "confirmations": [...],
          "suggested_corrections": {...},
          "ask_user_about": [...]
        }
        """
    )

    human_prompt = HumanMessage(
        f"""User Input: {user_input}

Extracted Fields: {extracted_character.model_dump_json()}

Please verify the correctness of these fields based on the user input."""
    )

    prompt_template = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0).with_structured_output(
        ReflectionFeedback,
        method="json_mode",
    )

    reflection_chain = prompt_template | llm

    # Invoke the reflection chain with the provided data
    reflection_result = reflection_chain.invoke(
        {"user_input": user_input, "extracted_character": extracted_character.model_dump()})
    return reflection_result


# Example usage of model_dump and model_dump_json:
if __name__ == "__main__":
    # Example usage:
    user_query = "Mi personaje de anime se llama Brais y tiene 28 años"
    # This might be the extracted character from the Field Mapper node:
    extracted_char = AnimeCharacter(
        name="Brais",  # Not explicitly mentioned
        age=28,  # User hints "about 40" but not certain
        gender=None,  # Not stated, possibly inferred as male by cultural context, but let's keep None
        physical_appearance=None,
        personality=None,
        # Possibly "helping villagers" implies kind/helpful, but let's say None to see if reflection suggests something
        abilities_power=None,  # Not mentioned any special power
        occupation=None  # Mentioned
    )

    # Run reflection
    reflection = reflect_on_extraction(user_query, extracted_char)

    # Print reflection results
    print("Reflection Feedback:")
    print(reflection.model_dump_json(indent=1))
