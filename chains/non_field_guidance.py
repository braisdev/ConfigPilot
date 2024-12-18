from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def provide_related_info(user_input: str):

    print("-- Providing related info --")

    system_message = SystemMessage(
        """
        You are a helpful assistant that it's main job is to always redirect the user to the main objective, which is
        fill the following necessary json fields to create an anime character:
        
        <anime-character-fields>
        - name (string)
        - age (integer)
        - gender (string)
        - physical_appearance (string)
        - personality (string)
        - abilities_power (string)
        - occupation (string)
        </anime-character-fields>
        
        The user reached you because his input was not related with the json fields. 
        
        Please redirect the conversation to the main objective, selecting **just one** of the fields to encourage
        the user to provide you information of the anime character that is looking to build.
        """
    )

    human_message = HumanMessage(
        f"""
        This is the user input that is not related with your objective:
        
        User input: {user_input}
        
        Please, kindly redirect him to the main objective, selecting **just one** of the fields to encourage
        """
    )

    prompt_components = [system_message, human_message]

    prompt_template = ChatPromptTemplate.from_messages(prompt_components)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

    non_field_guidance_chain = prompt_template | llm

    response = non_field_guidance_chain.invoke({"user_input": user_input})

    return response


if __name__ == "__main__":
    user_query = "I'm really tired"
    result = provide_related_info(user_input=user_query)
    print(result.content)


