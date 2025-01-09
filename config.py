openai_api_key: str = "REPLACE WITH YOUR OPENAI API KEY"
client_id: str = "REPLACE WITH YOUR PATSNAP CLIENT ID"
client_secret: str = "REPLACE WITH YOUR PATSNAP CLIENT SECRET"
date_range_string: str = " and PBD:[* TO 20241231]"
embedding_model_name: str = "Snowflake/snowflake-arctic-embed-xs"
persist_dir = "./chroma_db/vdb"
openai_model_name: str = "gpt-4o-2024-08-06"
temperature: float = 0.2
retriever_k: int = 5
openai_assistant_id: str = "REPLACE WITH YOUR OPENAI ASSISTANT ID"
openai_vector_store_id: str = "REPLACE WITH YOUR OPENAI VECTOR STORE ID"
sync_local_files_to_openai: bool = True

SYSTEM_PROMPT: str = (
    "You are a helpful product design assistant, and you will provide a list of potential novel solutions based on the requirements of user. Existing patents are provided for you to brainstorm and inspire new ideas, and you can query them through the PatentSearchTool. When searching, set the assignee to the renowned brands in the area so that patents returned can be more valuable."
)

ASSISTANT_INSTRUCTIONS: str = (
    "You are a helpful product design assistant, and you will provide a list of potential novel solutions based on the requirements of user. Existing patents are provided for you to brainstorm and inspire new ideas."
)

HUMAN_PROMPT: str = (
    "I need a new design for a hairdryer. The design should be innovative and unique. Can you provide me with some ideas?"
)

IMAGE_GEN_PROMPT: str = (
    "We have some innovative product design ideas. Generate a detailed prompt to generate an design image based on the following description: "
)