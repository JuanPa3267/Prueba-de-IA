import json
import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel # Correct import for BaseModel
from dotenv import load_dotenv
import mysql.connector # Import the MySQL connector
from typing import Any
import google.generativeai as genai # Import Google Gemini library

load_dotenv()

logger = logging.getLogger(__name__)

BACKEND_SERVER = os.getenv("SERVER_URL", "http://localhost:8000") # Default to localhost if not set

app = FastAPI(servers=[{"url": BACKEND_SERVER}])

# Configure Gemini with your API key
# Make sure OPEN_AI_API_KEY in your .env file is actually your GOOGLE_API_KEY
# It's better to use a specific name like GOOGLE_GEMINI_API_KEY
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
if not GOOGLE_GEMINI_API_KEY:
    raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# MySQL Database Configuration
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "1234"), # IMPORTANT: Change this!
    "database": os.getenv("MYSQL_DATABASE", "sakila"), # IMPORTANT: Change this to your DB name
}

def get_db_connection():
    """Establishes a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to MySQL: {err}")
        raise

def get_schema():
    """
    Retrieves the schema of the MySQL database.
    This is a basic implementation; for complex databases,
    you might want to include more details like foreign keys,
    indexes, etc.
    """
    schema_info = {}
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True) # Use dictionary=True to get column names

        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        for table in tables:
            table_name = list(table.values())[0] # Get the table name from the dictionary
            schema_info[table_name] = []
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            for col in columns:
                schema_info[table_name].append({
                    "column_name": col["Field"],
                    "data_type": col["Type"],
                    "nullable": col["Null"] == "YES"
                })
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error getting database schema: {e}")
        return {"error": str(e)}
    return json.dumps(schema_info, indent=2) # Return as JSON string

def query(sql_query: str):
    """
    Executes a SQL query against the MySQL database and returns the results.
    """
    results = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True) # Returns results as dictionaries
        cursor.execute(sql_query)
        # For SELECT queries, fetch results
        if sql_query.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
        # For INSERT, UPDATE, DELETE, commit changes
        else:
            conn.commit()
            results = {"message": "Query executed successfully."}
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        logger.error(f"Error executing SQL query: {err}")
        return {"error": str(err)}
    except Exception as e:
        logger.error(f"An unexpected error occurred during query: {e}")
        return {"error": str(e)}
    return results

async def human_query_to_sql(human_query: str):
    # Obtenemos el esquema de la base de datos
    database_schema = get_schema()
    if "error" in database_schema:
        return {"sql_query": None, "error": database_schema["error"]}

    system_message = f"""
    Dado el siguiente esquema de base de datos MySQL, escribe una consulta SQL que recupere la información solicitada.
    Devuelve la consulta SQL dentro de una estructura JSON con la clave "sql_query".
    Asegúrate de que la consulta SQL sea válida para MySQL.
    <example>{{
        "sql_query": "SELECT nombre, lenguaje_principal FROM desarrolladores WHERE experiencia_anios > 5;",
        "original_query": "Muéstrame los desarrolladores con más de 5 años de experiencia."
    }}
    </example>
    <schema>
    {database_schema}
    </schema>
    """
    user_message = human_query

    # Usamos genai.GenerativeModel para interactuar con Gemini
    model = genai.GenerativeModel(model_name="gemini-2.5-pro") # Using 1.5 Pro for better SQL generation
    try:
        response = await model.generate_content_async(
            contents=[
                {"role": "user", "parts": [system_message]},
                {"role": "model", "parts": ['```json\n{\n    "sql_query": "SELECT * FROM desarrolladores WHERE lenguaje_principal = \'Java\';",\n    "original_query": "Buscar desarrolladores que sepan Java"\n}\n```']}, # Example in the system message is not enough for few-shot
                {"role": "user", "parts": [user_message]}
            ]
        )
        # Gemini often returns content wrapped in markdown, try to extract the JSON
        text_content = response.text
        if text_content.startswith("```json"):
            text_content = text_content[len("```json"):].strip()
        if text_content.endswith("```"):
            text_content = text_content[:-len("```")].strip()
        
        return json.loads(text_content)
    except Exception as e:
        logger.error(f"Error generating SQL with Gemini: {e}")
        return {"sql_query": None, "error": str(e)}

async def build_answer(result: list[dict[str, Any]] | dict[str, Any], human_query: str) -> str | None:
    # Handle cases where query() might return an error dictionary
    if isinstance(result, dict) and "error" in result:
        return f"Lo siento, hubo un error al consultar la base de datos: {result['error']}"

    system_message = f"""
    Dada la pregunta de un usuario y la respuesta en formato JSON de la base de datos de la cual el usuario quiere obtener la respuesta,
    escribe una respuesta amigable y concisa para la pregunta del usuario.
    Si la consulta resultó en una actualización o eliminación, simplemente confirma la acción.
    Si no hay resultados, indícalo claramente.

    <user_question>
    {human_query}
    </user_question>
    <sql_response>
    {json.dumps(result, indent=2)}
    </sql_response>
    """

    model = genai.GenerativeModel(model_name="gemini-2.5-flash") # Using 2.5 Flash for better conversational responses
    try:
        response = await model.generate_content_async(
            contents=[
                {"role": "user", "parts": [system_message]}
            ]
        )
        return response.text
    except Exception as e:
        logger.error(f"Error building answer with Gemini: {e}")
        return f"Lo siento, no pude generar una respuesta. Error: {str(e)}"


class PostHumanQueryPayload(BaseModel):
    human_query: str


class PostHumanQueryResponse(BaseModel):
    answer: str


@app.post(
    "/human_query",
    name="Human Query",
    operation_id="post_human_query",
    description="Gets a natural language query, internally transforms it to a SQL query, queries the database, and returns the result.",
    response_model=PostHumanQueryResponse
)
async def human_query_endpoint(payload: PostHumanQueryPayload): # Renamed to avoid conflict with function
    # Transforma la pregunta a sentencia SQL
    sql_response = await human_query_to_sql(payload.human_query)

    if sql_response["sql_query"] is None:
        return {"answer": f"Falló la generación de la consulta SQL: {sql_response.get('error', 'Error desconocido')}"}

    sql_query = sql_response["sql_query"]
    original_query_from_llm = sql_response.get("original_query", payload.human_query)

    logger.info(f"Generated SQL: {sql_query}")

    # Hace la consulta a la base de datos
    result = query(sql_query) # query is not async, so no await needed here
    if isinstance(result, dict) and "error" in result:
        return {"answer": f"Falló la consulta a la base de datos: {result['error']}"}

    # Transforma la respuesta SQL a un formato más humano
    answer = await build_answer(result, original_query_from_llm)
    if not answer:
        return {"answer": "Falló la generación de la respuesta"}

    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    # Make sure to set your MySQL credentials in a .env file:
    # MYSQL_HOST=localhost
    # MYSQL_USER=root
    # MYSQL_PASSWORD=your_mysql_password
    # MYSQL_DATABASE=mi_db_desarrolladores
    # GOOGLE_GEMINI_API_KEY=YOUR_GEMINI_API_KEY

    uvicorn.run(app, host="0.0.0.0", port=8000)