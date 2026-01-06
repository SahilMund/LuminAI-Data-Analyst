from typing import List, Dict, Any
from sqlalchemy import create_engine, inspect, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.logging_config import get_logger
# from langchain_community.vectorstores import PGVector
from langchain_postgres.vectorstores import PGVector
from fastapi import HTTPException
from langchain_core.documents import Document
import pandas as pd
from app.config.env import (DATABASE_URL)
from typing import List, Optional

logger = get_logger(__name__)


class DB:
    def __init__(self, db_url: str):
        """
        Initialize the database connection.

        Args:
            db_url (str): Database URL
        """
        self.engine = create_engine(db_url)
        self.session = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine)
        self.inspector = inspect(self.engine)

    def execute_query(self, query: str) -> list:
        print(f"DEBUG_SQL: Executing Query: {query}")
        with self.session() as session:
            result = session.execute(text(query))
            if result.returns_rows:
                rows = [row for row in result.fetchall()]
                print(f"DEBUG_SQL: Query returned {len(rows)} rows")
                return rows
            else:
                session.commit()
                print("DEBUG_SQL: Query executed successfully (no rows returned)")
                return []

    def create_session(self) -> Session:
        return self.session()

    def get_schemas(self, table_names: List[str]) -> List[Dict]:
        try:
            # Create an inspector object
            inspector = inspect(self.engine)

            # Initialize an array to hold the schema information for all tables
            schemas_info = []

            for table_name in table_names:
                schema_info = {
                    "table_name": table_name,
                    "schema": []
                }

                # Get the columns for the specified table
                columns = inspector.get_columns(table_name)
                # Collect column information
                for column in columns:
                    schema_info["schema"].append({
                        "name": column['name'],
                        "type": str(column['type']),
                        "nullable": column['nullable']
                    })

                # Append the schema information for the current table to the list
                schemas_info.append(schema_info)

            # Return the schema information for all tables
            return schemas_info

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return []  # Return an empty list in case of an error

    async def insert_dataframe(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Insert pandas DataFrame into database"""
        try:
            with self.session() as session:
                df.to_sql(
                    name=table_name,
                    con=session.get_bind(),
                    if_exists='replace',
                    index=False
                )
                return {
                    "message": f"Successfully inserted data into table {table_name}",
                    "rows_processed": len(df)
                }
        except Exception as e:
            logger.error(f"Data insertion error: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to insert data into database")


class VectorDB:
    def __init__(self):
        """Initialize VectorDB with connection string"""
        self.connection_string = DATABASE_URL
        self._embedding: Optional[HuggingFaceEmbeddings] = None

    def initialize_embedding(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        """
        if self._embedding is None:
            logger.info(f"Initializing HuggingFaceEmbeddings with model: {model_name}")
            self._embedding = HuggingFaceEmbeddings(model_name=model_name)
            return "Embedding model initialized successfully."
        return "Embedding model already initialized."

    @property
    def embeddings(self):
        if self._embedding is None:
            raise ValueError(
                "Embedding model not initialized. Call initialize_embedding() first.")
        return self._embedding

    async def insert_data(self, documents: List[Document], collection_name: str) -> PGVector:
        """Insert documents into vector store"""
        try:
            return PGVector.from_documents(
                embeddings=self.embeddings,
                documents=documents,
                collection_name=collection_name,
                connection=self.connection_string,
                use_jsonb=True,
            )
        except Exception as e:
            logger.exception(f"Vector store insertion error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to insert documents into vector store: {str(e)}")

    def get_vector_store(self, collection_name: str) -> PGVector:
        """Get existing vector store"""
        try:
            return PGVector(
                connection=self.connection_string,
                embeddings=self.embeddings,
                collection_name=collection_name,
                use_jsonb=True,
            )
        except Exception as e:
            logger.exception(f"Vector store retrieval error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve vector store: {str(e)}")
