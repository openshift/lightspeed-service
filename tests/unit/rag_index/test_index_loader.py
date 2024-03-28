"""Unit test for the index loader module."""

from unittest.mock import patch

from ols.app.models.config import ReferenceContent
from ols.src.rag_index.index_loader import IndexLoader
from ols.utils import config
from tests.mock_classes.mock_llama_index import MockLlamaIndex


def test_index_loader_empty_config(caplog):
    """Test index loader with empty/None config."""
    index_loader_obj = IndexLoader(None)
    index = index_loader_obj.vector_index

    assert "required parameters are not set" in caplog.text
    assert not hasattr(index_loader_obj, "_embed_model")
    assert index is None


@patch("ols.src.rag_index.index_loader.ServiceContext.from_defaults")
@patch("ols.src.rag_index.index_loader.StorageContext.from_defaults")
def test_index_loader_no_id(storage_context, service_context):
    """Test index loader without index id."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./some_dir"

    index_loader_obj = IndexLoader(config.ols_config.reference_content)
    index = index_loader_obj.vector_index

    assert (
        index_loader_obj._embed_model == "local:sentence-transformers/all-mpnet-base-v2"
    )
    assert index is None


@patch("ols.src.rag_index.index_loader.ServiceContext.from_defaults")
@patch("ols.src.rag_index.index_loader.StorageContext.from_defaults")
@patch("llama_index.vector_stores.faiss.FaissVectorStore.from_persist_dir")
@patch("ols.src.rag_index.index_loader.load_index_from_storage", new=MockLlamaIndex)
def test_index_loader(storage_context, service_context, from_persist_dir):
    """Test index loader."""
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./some_dir"
    config.ols_config.reference_content.product_docs_index_id = "./some_id"

    from_persist_dir.return_value = None

    index_loader_obj = IndexLoader(config.ols_config.reference_content)
    index = index_loader_obj.vector_index

    assert isinstance(index, MockLlamaIndex)
