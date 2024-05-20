"""Unit test for the index loader module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from ols import config
from ols.app.models.config import ReferenceContent
from ols.src.rag_index.index_loader import IndexLoader
from tests.mock_classes.mock_llama_index import MockLlamaIndex


@pytest.fixture
def ols_configuration():
    """Fixture with empty configuration."""
    config.reload_empty()
    assert config.ols_config is not None
    return config.ols_config


def test_index_loader_empty_config(caplog):
    """Test index loader with empty/None config."""
    index_loader_obj = IndexLoader(None)
    index = index_loader_obj.vector_index

    assert "required parameters are not set" in caplog.text
    assert not hasattr(index_loader_obj, "_embed_model")
    assert index is None


@patch("ols.src.rag_index.index_loader.StorageContext.from_defaults")
def test_index_loader_no_id(storage_context, ols_configuration):
    """Test index loader without index id."""
    ols_configuration.reference_content = ReferenceContent(None)
    ols_configuration.reference_content.product_docs_index_path = Path("./some_dir")

    index_loader_obj = IndexLoader(ols_configuration.reference_content)
    index = index_loader_obj.vector_index

    assert (
        index_loader_obj._embed_model == "local:sentence-transformers/all-mpnet-base-v2"
    )
    assert index is None


@patch("ols.src.rag_index.index_loader.StorageContext.from_defaults")
@patch("llama_index.vector_stores.faiss.FaissVectorStore.from_persist_dir")
@patch("ols.src.rag_index.index_loader.load_index_from_storage", new=MockLlamaIndex)
def test_index_loader(storage_context, from_persist_dir, ols_configuration):
    """Test index loader."""
    ols_configuration.reference_content = ReferenceContent(None)
    ols_configuration.reference_content.product_docs_index_path = Path("./some_dir")
    ols_configuration.reference_content.product_docs_index_id = "./some_id"

    from_persist_dir.return_value = None

    index_loader_obj = IndexLoader(ols_configuration.reference_content)
    index = index_loader_obj.vector_index

    assert isinstance(index, MockLlamaIndex)
