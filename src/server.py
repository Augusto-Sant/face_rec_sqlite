# app.py
import os
import math
from deepface import DeepFace
import pandas as pd
import numpy as np
from tqdm import tqdm
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    LargeBinary,
    DECIMAL,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker, Session, relationship

# Create an engine for the SQLite database
engine = create_engine("sqlite:///facialdb.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Declare a base for your classes
Base = declarative_base()


# Define the face_meta table
class FaceMeta(Base):
    __tablename__ = "face_meta"

    ID = Column(Integer, primary_key=True)
    IMG_NAME = Column(String(10))
    CLASSE = Column(String)
    MODULO = Column(String)
    SUB_CLASSE = Column(String)
    EMBEDDING = Column(LargeBinary)  # Equivalent to BLOB


# Define the face_embeddings table
class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    ID = Column(Integer, primary_key=True)
    FACE_ID = Column(Integer, ForeignKey("face_meta.ID"))
    DIMENSION = Column(Integer)
    VALUE = Column(DECIMAL(5, 30))

    # Establish a relationship between FaceMeta and FaceEmbedding (Optional)
    face_meta = relationship("FaceMeta", backref="embeddings")


app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ImageRequest(BaseModel):
    image: str  # The base64 string


@app.post("/rec")
async def rec_endpoint(
    image_data: ImageRequest,
    db_session: Session = Depends(get_db),
):
    # Decode the Base64 image
    encoded = image_data.image  # Split the data URL

    # Generate the embedding for the target image using the Facenet model
    target_embedding = DeepFace.represent(
        img_path=encoded,
        model_name="Facenet",
    )[
        0
    ]["embedding"]

    # Build the target_statement dynamically
    target_statement = ""
    for i, value in enumerate(target_embedding):
        target_statement += (
            f"SELECT {i} AS dimension, {value} AS value"  # sqlite version
        )
        if i < len(target_embedding) - 1:
            target_statement += " UNION ALL "

    # Build the main SQL query with the dynamic target_statement
    classe_value = "CLASSE_TESTE"
    modulo_value = "MODULO_TESTE"
    sub_classe_value = "SUB_CLASSE_TESTE"

    select_statement = f"""
        SELECT *
        FROM (
            SELECT img_name, SUM(subtract_dims) AS distance_squared
            FROM (
                SELECT face_meta.img_name,
                    (face_embeddings.value - target.value) * (face_embeddings.value - target.value) AS subtract_dims
                FROM face_meta
                LEFT JOIN face_embeddings ON face_meta.id = face_embeddings.face_id
                LEFT JOIN (
                    {target_statement}
                ) target ON face_embeddings.dimension = target.dimension
                WHERE face_meta.classe = '{classe_value}'
                    AND face_meta.modulo = '{modulo_value}'
                    AND face_meta.sub_classe = '{sub_classe_value}'
            )
            GROUP BY img_name
        )
        WHERE distance_squared < 100
        ORDER BY distance_squared ASC
    """

    # Execute the query
    results = db_session.execute(text(select_statement))
    instances = []
    for result in results:
        img_name = result[0]
        distance_squared = result[1]

        # Calculate the square root of the distance
        instance = [img_name, math.sqrt(distance_squared)]
        instances.append(instance)
    # Create a DataFrame from the instances list
    result_df = pd.DataFrame(instances, columns=["img_name", "distance"])
    result_json = result_df.to_dict(
        orient="records"
    )  # Converts each row to a dictionary
    return JSONResponse(content=result_json)


@app.post("/update")
def update_model(db_session: Session = Depends(get_db)):
    Base.metadata.drop_all(bind=engine)  # Drop all tables
    Base.metadata.create_all(bind=engine)  # Recreate all tables

    facial_img_paths = []
    for root, directory, files in os.walk("dataset"):
        for file in files:
            if file.endswith(".jpg"):
                facial_img_paths.append(os.path.join(root, file))
    instances = []
    for facial_img_path in tqdm(facial_img_paths):
        embedding = DeepFace.represent(
            img_path=facial_img_path,
            model_name="Facenet",
        )[
            0
        ]["embedding"]

        # store
        instance = [facial_img_path, embedding]
        instances.append(instance)

    df = pd.DataFrame(instances, columns=["img_name", "embedding"])

    # Loop through the DataFrame rows
    for index, instance in tqdm(df.iterrows(), total=df.shape[0]):
        img_name = instance["img_name"]
        embeddings = instance["embedding"]
        embeddings = np.array(instance["embedding"])

        # Insert into face_meta
        if img_name == "dataset/img2.jpg":
            face_meta_entry = FaceMeta(
                ID=index,
                IMG_NAME=img_name,
                EMBEDDING=embeddings.tobytes(),
                CLASSE="WHAT",
                MODULO="WHAT2",
                SUB_CLASSE="WHAT3",
            )
        else:
            face_meta_entry = FaceMeta(
                ID=index,
                IMG_NAME=img_name,
                EMBEDDING=embeddings.tobytes(),
                CLASSE="CLASSE_TESTE",
                MODULO="MODULO_TESTE",
                SUB_CLASSE="SUB_CLASSE_TESTE",
            )
        db_session.add(face_meta_entry)

        # Insert into face_embeddings for each embedding dimension
        for i, embedding in enumerate(embeddings):
            face_embedding_entry = FaceEmbedding(
                FACE_ID=index, DIMENSION=i, VALUE=embedding
            )
            db_session.add(face_embedding_entry)

    # Commit the changes to the database
    db_session.commit()


def start_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000)
