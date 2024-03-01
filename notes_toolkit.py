import os
from typing import Type, Optional, List
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

# Encompass your custom tools in a toolkit class
class NotesToolkit():

    # Class to define a tool that creates notes
    class CreateNote(BaseTool):
        # Class to define the input parameters and instructions for agent
        class CreateNoteInput(BaseModel):
            # Parameters and their descriptions for agent
            note: str = Field(description="This should be the note that needs to be saved.")
            path: str = Field(description="""
                Should be a path from the system message. 
                It is wrapped with '<' and the wrap ends with '>'.
                Remove the '<' and '>' from the string before setting parameter. 
                For example: './path/', not '<./path/>.
            """)

            # Validation method to check parameter input from agent
            @validator("note")
            def validate_note_param(note):
                if not note:
                    raise ValueError("CreateNote tool error: Note parameter is empty")
                else:
                    return note
                
            # Validation method to check parameter input from agent
            @validator("path")
            def validate_path_param(path):
                if path[0] == '<':
                    path = path[1:]
                if path[-1] == '>':
                    path = path[:-1]
                if os.path.exists(path):
                    return path
                else:
                    raise ValueError("CreateNote tool error: Path does not exist")

        # Tool name, description, and argument schema (defined by input subclass)
        name = "create_note"
        description = f"Useful for when you want to save a note for the user."
        args_schema: Type[BaseModel] = CreateNoteInput

        # Actual method to be run when agent calls the tool
        def _run(self, note: str, path: str) -> str:
            """Use the tool."""
            with open(f"{path}notes.txt", 'a') as f:
                f.writelines([note + "\n"])
            return "Success"
    
    # Class to define a tool that queries existing notes
    class QueryNotes(BaseTool):
        # Class to define the input parameters and instructions for agent
        class QueryNotesInput(BaseModel):
            # Parameters and their descriptions for agent
            query: str = Field(description="This should be the query for searching through the notes with.")
            path: str = Field(description="""
                Should be a path from the system message. 
                It is wrapped with '<' and the wrap ends with '>'.
                Remove the '<' and '>' from the string before setting parameter. 
                For example: './path/', not '<./path/>.
            """)

            # Validation method to check parameter input from agent
            @validator("query")
            def validate_query_param(query):
                if not query:
                    raise ValueError("QueryNotes tool error: Query parameter is empty")
                else:
                    return query
            
            # Validation method to check parameter input from agent
            @validator("path")
            def validate_path_param(path):
                print(path)
                if path[0] == '<':
                    path = path[1:]
                if path[-1] == '>':
                    path = path[:-1]
                if os.path.exists(path):
                    return path
                else:
                    raise ValueError("QueryNotes tool error: Path does not exist")

        # Tool name, description, and argument schema (defined by input subclass)
        name = "query_notes"
        description = f"Useful for when you want to query existing notes for more information."
        args_schema: Type[BaseModel] = QueryNotesInput

        # Actual method to be run when agent calls the tool
        def _run(self, query: str, path: str) -> str:
            """Use the tool."""
            loader = TextLoader(f"{path}notes.txt")
            raw_docs = loader.load()
            text_splitter = CharacterTextSplitter()
            split_docs = text_splitter.split_documents(raw_docs)
            db = Chroma.from_documents(split_docs, OpenAIEmbeddings())
            relevant_docs = db.similarity_search(query, k=1)
            return relevant_docs
    
    # Init above tools and make available
    def __init__(self) -> None:
        self.tools = [
            self.CreateNote(),
            self.QueryNotes(),
        ]

    # Method to get tools (for ease of use, made so class works similarly to LangChain toolkits)
    def get_tools(self) -> List[BaseTool]:
        return self.tools