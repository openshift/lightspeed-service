"""Unit test for the index loader module."""

from pathlib import Path
from unittest.mock import patch

from ols import config
from ols.app.models.config import PostgresConfig, ReferenceContent
from ols.constants import VectorStoreType
from ols.src.rag_index.index_loader import IndexLoader
from tests.mock_classes.mock_llama_index import MockLlamaIndex


def test_index_loader_empty_config(caplog):
    """Test index loader with empty/None config."""
    index_loader_obj = IndexLoader(None)
    index = index_loader_obj.vector_index

    assert "required parameters are not set" in caplog.text
    assert not hasattr(index_loader_obj, "_embed_model")
    assert index is None


def test_index_loader_no_id():
    """Test index loader without index id."""
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = Path("./some_dir")

    with patch("llama_index.core.StorageContext.from_defaults"):
        index_loader_obj = IndexLoader(config.ols_config.reference_content)
        index = index_loader_obj.vector_index

        assert (
            index_loader_obj._embed_model
            == "local:sentence-transformers/all-mpnet-base-v2"
        )
        assert index is None


def test_index_loader():
    """Test index loader."""
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = Path("./some_dir")
    config.ols_config.reference_content.product_docs_index_id = "./some_id"

    with (
        patch("llama_index.core.StorageContext.from_defaults"),
        patch(
            "llama_index.vector_stores.faiss.FaissVectorStore.from_persist_dir"
        ) as from_persist_dir,
        patch("llama_index.core.load_index_from_storage", new=MockLlamaIndex),
    ):
        from_persist_dir.return_value = None

        index_loader_obj = IndexLoader(config.ols_config.reference_content)
        index = index_loader_obj.vector_index

        assert isinstance(index, MockLlamaIndex)


def test_index_loader_from_faiss():
    """Test index loader when 'faiss' is selected for the vector store type."""
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.vector_store_type = VectorStoreType.FAISS
    config.ols_config.reference_content.product_docs_index_path = Path("./some_dir")
    config.ols_config.reference_content.product_docs_index_id = "./some_id"

    with (
        patch("llama_index.core.StorageContext.from_defaults"),
        patch(
            "llama_index.vector_stores.faiss.FaissVectorStore.from_persist_dir"
        ) as from_persist_dir,
        patch("llama_index.core.load_index_from_storage", new=MockLlamaIndex),
    ):
        from_persist_dir.return_value = None

        index_loader_obj = IndexLoader(config.ols_config.reference_content)
        index = index_loader_obj.vector_index

        assert isinstance(index, MockLlamaIndex)


def test_index_loader_from_postgres():
    """Test index loader when 'postgres' is selected for the vector store type."""
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.vector_store_type = VectorStoreType.POSTGRES
    config.ols_config.reference_content.product_docs_index_id = "some_id"
    config.ols_config.reference_content.postgres = PostgresConfig()

    with (
        patch("llama_index.core.StorageContext.from_defaults"),
        patch(
            "llama_index.vector_stores.postgres.PGVectorStore.from_params"
        ) as from_params,
        patch(
            "llama_index.core.VectorStoreIndex.from_vector_store", new=MockLlamaIndex
        ),
    ):
        from_params.return_value = None

        index_loader_obj = IndexLoader(config.ols_config.reference_content)
        index = index_loader_obj.vector_index

        assert isinstance(index, MockLlamaIndex)
