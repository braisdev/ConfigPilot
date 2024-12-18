from state import GraphState
import psycopg2

def persist_character_fields(state: GraphState):
    # Extract the anime_character data from the state
    anime_character = state.get("anime_character", {})

    name = anime_character.get("name")
    age = anime_character.get("age")
    gender = anime_character.get("gender")
    physical_appearance = anime_character.get("physical_appearance")
    personality = anime_character.get("personality")
    abilities_power = anime_character.get("abilities_power")
    occupation = anime_character.get("occupation")

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        host="langgraph-postgres",
        port="5432",
        database="my_characters",
        user="postgres",
        password="postgres"
    )

    try:
        with conn:
            with conn.cursor() as cur:
                # Create the table if it doesn't exist
                create_table_query = """
                CREATE TABLE IF NOT EXISTS anime_characters (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    age INT,
                    gender TEXT,
                    physical_appearance TEXT,
                    personality TEXT,
                    abilities_power TEXT,
                    occupation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cur.execute(create_table_query)

                insert_query = """
                    INSERT INTO anime_characters
                    (name, age, gender, physical_appearance, personality, abilities_power, occupation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cur.execute(
                    insert_query,
                    (name, age, gender, physical_appearance, personality, abilities_power, occupation)
                )
    except Exception as e:
        print(f"Error inserting character: {e}")
    finally:
        conn.close()

    # Return a dictionary indicating persistence success
    return {"persisted": True}
