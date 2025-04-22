"""Unit test for the index loader module."""

from unittest.mock import patch

from ols import config
from ols.app.models.config import ReferenceContent, ReferenceContentIndex
from ols.src.rag_index.index_loader import IndexLoader
from tests.mock_classes.mock_llama_index import MockLlamaIndex


def test_index_loader_empty_config(caplog):
    """Test index loader with empty/None config."""
    index_loader_obj = IndexLoader(None)
    indexes = index_loader_obj.vector_indexes

    assert "required parameters are not set" in caplog.text
    assert not hasattr(index_loader_obj, "_embed_model")
    assert indexes is None


def test_index_loader_no_id():
    """Test index loader without index id."""
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.indexes = [
        ReferenceContentIndex(
            {"product_docs_index_path": "./some_dir", "product_docs_index_id": None}
        )
    ]

    with patch("llama_index.core.StorageContext.from_defaults"):
        index_loader_obj = IndexLoader(config.ols_config.reference_content)
        indexes = index_loader_obj.vector_indexes

        assert (
            index_loader_obj._embed_model
            == "local:sentence-transformers/all-mpnet-base-v2"
        )
        assert indexes is None


def test_index_loader():
    """Test index loader."""
    config.ols_config.reference_content = ReferenceContent(None)

    with (
        patch("llama_index.core.StorageContext.from_defaults"),
        patch(
            "llama_index.vector_stores.faiss.FaissVectorStore.from_persist_dir"
        ) as from_persist_dir,
        patch("llama_index.core.load_index_from_storage", new=MockLlamaIndex),
    ):
        config.ols_config.reference_content.indexes = [
            ReferenceContentIndex(
                {
                    "product_docs_index_path": "./some_dir",
                    "product_docs_index_id": "./some_id",
                }
            )
        ]
        from_persist_dir.return_value = None

        index_loader_obj = IndexLoader(config.ols_config.reference_content)
        indexes = index_loader_obj.vector_indexes

        assert len(indexes) == 1
        assert isinstance(indexes[0], MockLlamaIndex)
