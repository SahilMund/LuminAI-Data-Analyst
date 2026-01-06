from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select, func
from io import BytesIO
import pandas as pd
from app.config.logging_config import get_logger
from app.config.db_config import DB
from app.api.db.data_sources import DataSources
from app.utils.reader_utils import (pdf_to_document, text_to_document)
from app.config.db_config import VectorDB
import uuid
from app.api.validators.data_source_validator import (
    GetSourceTable, AddDataSource)
from app.utils.response_utils import create_response

# Set up logging
logger = get_logger(__name__)
vector_db = VectorDB()


async def upload_spreadsheet(id: int, file: UploadFile, db: DB) -> JSONResponse:
    buffer = None
    try:
        logger.info(f"Processing file: {file.filename}")
        session = db.create_session()
        # Validate file extension
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            return JSONResponse(status_code=400, content=create_response(
                status_code=400,
                message="Only CSV and Excel files are allowed",
                data={}
            ))

        # Read file contents
        contents = await file.read()
        buffer = BytesIO(contents)

        # Determine file type and read accordingly
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(buffer)
        else:
            df = pd.read_excel(buffer)

        # Convert all column names to lowercase and replace spaces with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Generate a unique table name
        base_name = file.filename.rsplit('.', 1)[0].lower()
        table_name = f"{base_name}_{uuid.uuid4().hex[:8]}"

        # Insert data into database
        rows_affected = await db.insert_dataframe(df, table_name)

        # Create DataSources entry
        new_data_source = DataSources(
            name=file.filename,
            type='spreadsheet',
            table_name=table_name,
            user_id=id
        )

        session.add(new_data_source)
        session.commit()
        session.refresh(new_data_source)

        logger.info(f"Successfully processed file. Rows: {rows_affected}")
        return JSONResponse(status_code=201, content=create_response(
            status_code=201,
            message="Data uploaded successfully",
            data={
                "table_name": table_name,
                "rows_processed": rows_affected,
                "data_source_id": new_data_source.id
            }
        ))

    except HTTPException as he:
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="Something went wrong",
            data={"error": str(he)}
        ))
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="Database error occurred",
            data={"error": str(e)}
        ))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="An unexpected error occurred",
            data={"error": str(e)}
        ))


    finally:
        if buffer:
            buffer.close()
        logger.info("Upload process completed")


async def upload_document(id: int, file: UploadFile, db: DB) -> JSONResponse:
    buffer = None
    try:
        logger.info(f"Processing file: {file.filename}")
        session = db.create_session()
        vector_db.initialize_embedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Validate file extension
        if not file.filename.lower().endswith(('.pdf', '.doc', '.txt')):
            return JSONResponse(status_code=400, content=create_response(
                status_code=400,
                message="Only Pdf, Doc and text files are allowed",
                data={}
            ))
        # Generate a unique table name
        base_name = file.filename.rsplit('.', 1)[0].lower()
        table_name = f"{base_name}_{uuid.uuid4().hex[:8]}"

        # Read file contents
        contents = await file.read()
        buffer = BytesIO(contents)

        if file.filename.lower().endswith((".pdf")):
            documents = pdf_to_document(buffer, table_name)
        elif file.filename.lower().endswith(('.doc', '.txt')):
            documents = text_to_document(buffer, table_name)

        print(documents)
        await vector_db.insert_data(documents, table_name)
        # Create DataSources entry
        new_data_source = DataSources(
            name=file.filename,
            type='document',
            table_name=table_name,
            user_id=id
        )

        session.add(new_data_source)
        session.commit()
        session.refresh(new_data_source)

        return JSONResponse(status_code=201, content=create_response(
            status_code=201,
            message="Data uploaded successfully",
            data={
                "table_name": table_name,
                "data_source_id": new_data_source.id
            },
        ))

    except HTTPException as he:
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="Something went wrong",
            data={"error": str(he)}
        ))
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="Database error occurred",
            data={"error": str(e)}
        ))
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="An unexpected error occurred",
            data={"error": str(e)}
        ))
    finally:
        if buffer:
            buffer.close()
        logger.info("Upload process completed")


async def add_datasource(data: AddDataSource, id: int, db: DB) -> JSONResponse:
    try:
        session = db.create_session()
        # Create DataSources entry
        new_data_source = DataSources(
            name=data.table_name,
            type='url',
            connection_url=data.source_name,
            user_id=id
        )

        session.add(new_data_source)
        session.commit()
        session.refresh(new_data_source)

        return JSONResponse(status_code=201, content=create_response(
            status_code=201,
            message="Data uploaded successfully",
            data={
                "table_name": data.table_name,
                "connection_url": data.source_name,
                "id": new_data_source.id
            }
        ))

    except HTTPException as he:
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="Something went wrong",
            data={"error": str(he)}
        ))
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="Database error occurred",
            data={"error": str(e)}
        ))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="An unexpected error occurred",
            data={"error": str(e)}
        ))



async def get_data_source_list(id: int, db: DB) -> JSONResponse:
    try:
        with db.session() as session:
            query = select(
                DataSources.id,
                DataSources.name,
                DataSources.type,
                DataSources.connection_url,
                DataSources.table_name,
                func.to_char(DataSources.created_at,
                             'YYYY-MM-DD').label('created_at')
            ).where(DataSources.user_id == id)

            result = session.execute(query)
            data_sources = result.mappings().all()

        # Convert to list of dicts and return
        sources = [dict(row) for row in data_sources]

        return JSONResponse(status_code=200, content=create_response(
            status_code=200,
            message="Data sources fetched successfully",
            data={"data_sources":sources}
        ))

    except HTTPException as he:
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="Something went wrong",
            data={"error": str(he)}
        ))
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="Database error occurred",
            data={"error": str(e)}
        ))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="An unexpected error occurred",
            data={"error": str(e)}
        ))


async def get_source_tables(source: GetSourceTable) -> JSONResponse:
    try:
        db = DB(source.db_url)
        tables = db.inspector.get_table_names()
        return JSONResponse(status_code=200, content=create_response(
            status_code=200,
            message="Tables fetched successfully",
            data={"tables":tables}
        ))
    except HTTPException as he:
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="Something went wrong",
            data={"error": str(he)}
        ))
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="Database error occurred",
            data={"error": str(e)}
        ))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content=create_response(
            status_code=500,
            message="An unexpected error occurred",
            data={"error": str(e)}
        ))
    finally:
        if 'engine' in locals():
            db.engine.dispose()
