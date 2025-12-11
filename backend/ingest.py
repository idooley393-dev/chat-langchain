"""Load Joint Publications PDFs from disk, split, and ingest into Weaviate."""

import logging
import os

import weaviate
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.indexes import SQLRecordManager, index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore

from backend.constants import WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME
from backend.embeddings import get_embeddings_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]


#########################
# Joint Publications from local PDFs
#########################


def load_joint_publications_from_disk():
    """
    Load all PDF files under backend/data (relative to this file).
    Put your Joint Publications (JP 1, JP 3-0, JP 5-0, etc.) as PDFs in that folder.
    """
    # Path to the folder containing your Joint Publication PDFs
    docs_dir = os.path.join(os.path.dirname(__file__), "data")

    logger.info(f"Loading Joint Publications from: {docs_dir}")

    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",        # recursively load all PDFs
        loader_cls=PyPDFLoader, # use PyPDFLoader to read PDFs
    )
    docs = loader.load()

    # Add helpful metadata such as fileName
    for d in docs:
        source = d.metadata.get("source", "")
        filename = os.path.basename(source) if source else ""
        d.metadata.setdefault("fileName", filename)

    logger.info(f"Loaded {len(docs)} documents from disk.")
    return docs


def ingest_general_guides_and_tutorials():
    """
    For your use case, 'general guides and tutorials' now means
    'all Joint Publication PDFs loaded from disk'.
    """
    return load_joint_publications_from_disk()


def ingest_docs():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )
    embedding = get_embeddings_model()

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    ) as weaviate_client:
        # Vector store for Joint Publications (re-using the existing index name)
        general_guides_and_tutorials_vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
            text_key="text",
            embedding=embedding,
            attributes=["source", "title"],
        )

        # Record manager to keep track of what we've indexed
        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME}",
            db_url=RECORD_MANAGER_DB_URL,
        )
        record_manager.create_schema()

        # Load your Joint Publications
        general_guides_and_tutorials_docs = ingest_general_guides_and_tutorials()

        # Split into chunks
        docs_transformed = text_splitter.split_documents(
            general_guides_and_tutorials_docs
        )
        docs_transformed = [
            doc for doc in docs_transformed if len(doc.page_content) > 10
        ]

        # Ensure 'source' and 'title' are always present (Weaviate requires them)
        for doc in docs_transformed:
            if "source" not in doc.metadata:
                doc.metadata["source"] = ""

            # If there's no title, fall back to filename or a generic label
            if "title" not in doc.metadata or not doc.metadata["title"]:
                source = doc.metadata.get("source", "")
                filename = os.path.basename(source) if source else ""
                doc.metadata["title"] = filename or "Joint Publication"

        # Index into Weaviate with record management
        indexing_stats = index(
            docs_transformed,
            record_manager,
            general_guides_and_tutorials_vectorstore,
            cleanup="full",
            source_id_key="source",
            force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
        )
        logger.info(f"Indexing stats: {indexing_stats}")

        # Log how many vectors are in the collection now
        num_vecs = (
            weaviate_client.collections.get(
                WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME
            )
            .aggregate.over_all()
            .total_count
        )
        logger.info(
            f"General Guides and Tutorials (Joint Publications) now has this many vectors: {num_vecs}",
        )


if __name__ == "__main__":
    ingest_docs()
