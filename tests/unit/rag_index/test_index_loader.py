"""Unit test for the index loader module."""

from pathlib import Path
from unittest.mock import patch

from ols import config
from ols.app.models.config import ReferenceContent
from ols.src.rag_index.index_loader import IndexLoader
from tests.mock_classes.mock_llama_index import MockLlamaIndex


def test_index_loader_empty_config(caplog):
    """Test index loader with empty/None config."""
    index_loader_obj = IndexLoader(None)
    index = index_loader_obj.vector_index

    assert "required parameters are not set" in caplog.text
    assert not hasattr(index_loader_obj, "_embed_model")
    assert index is None


@patch("llama_index.core.StorageContext.from_defaults")
def test_index_loader_no_id(storage_context):
    """Test index loader without index id."""
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = Path("./some_dir")

    index_loader_obj = IndexLoader(config.ols_config.reference_content)
    index = index_loader_obj.vector_index

    assert (
        index_loader_obj._embed_model == "local:sentence-transformers/all-mpnet-base-v2"
    )
    assert index is None


@patch("llama_index.core.StorageContext.from_defaults")
@patch("llama_index.vector_stores.faiss.FaissVectorStore.from_persist_dir")
@patch("llama_index.core.load_index_from_storage", new=MockLlamaIndex)
def test_index_loader(storage_context, from_persist_dir):
    """Test index loader."""
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = Path("./some_dir")
    config.ols_config.reference_content.product_docs_index_id = "./some_id"

    from_persist_dir.return_value = None

    index_loader_obj = IndexLoader(config.ols_config.reference_content)
    index = index_loader_obj.vector_index

    assert isinstance(index, MockLlamaIndex)
