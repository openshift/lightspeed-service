"""Unit test for the index loader module."""

import os
from unittest.mock import patch

import ols.src.rag_index.index_loader as il
from ols import config
from ols.app.models.config import ReferenceContent, ReferenceContentIndex
from tests.mock_classes.mock_llama_index import MockLlamaIndex
from tests.mock_classes.mock_retrievers import MockRetriever


def test_index_loader_empty_config(caplog):
    """Test index loader with empty/None config."""
    index_loader_obj = il.IndexLoader(None)
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

    with (
        patch("llama_index.core.StorageContext.from_defaults"),
        patch.dict(os.environ, {"TRANSFORMERS_CACHE": "", "TRANSFORMERS_OFFLINE": ""}),
    ):
        index_loader_obj = il.IndexLoader(config.ols_config.reference_content)
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
        patch.dict(os.environ, {"TRANSFORMERS_CACHE": "", "TRANSFORMERS_OFFLINE": ""}),
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

        index_loader_obj = il.IndexLoader(config.ols_config.reference_content)
        indexes = index_loader_obj.vector_indexes

        assert len(indexes) == 1
        assert isinstance(indexes[0], MockLlamaIndex)


def test_custom_weight_function():
    """Test custom weight function."""
    # Load llamaindex imports
    from llama_index.core.schema import NodeWithScore, TextNode

    il.load_llama_index_deps()

    # Mock retrieved nodes for a query from 3 indexes (2 chunks each)
    mock_retrieved_result = {
        ("query_text", 0): [
            NodeWithScore(node=TextNode(text="chunk1_index1"), score=0.75),
            NodeWithScore(node=TextNode(text="chunk2_index1"), score=0.73),
        ],
        ("query_text", 1): [
            NodeWithScore(node=TextNode(text="chunk1_index2"), score=0.755),
            NodeWithScore(node=TextNode(text="chunk2_index2"), score=0.735),
        ],
        ("query_text", 2): [
            NodeWithScore(node=TextNode(text="chunk1_index3"), score=0.745),
            NodeWithScore(node=TextNode(text="chunk2_index3"), score=0.738),
        ],
    }

    il.Settings.llm = il.resolve_llm(None)

    actual_fusion_retriever = il.QueryFusionRetriever(
        # Pass 3 mock retrievers, rest all fields are optional.
        retrievers=[MockRetriever] * 3,
        mode="simple",
    )
    sorted_result = actual_fusion_retriever._simple_fusion(mock_retrieved_result)
    assert len(sorted_result) == 6

    assert sorted_result[0].get_content() == "chunk1_index2"
    assert sorted_result[0].score == 0.755
    assert sorted_result[1].get_content() == "chunk1_index1"
    assert sorted_result[1].score == 0.75
    assert sorted_result[2].get_content() == "chunk1_index3"
    assert sorted_result[2].score == 0.745
    assert sorted_result[3].get_content() == "chunk2_index3"
    assert sorted_result[3].score == 0.738
    assert sorted_result[4].get_content() == "chunk2_index2"
    assert sorted_result[4].score == 0.735
    assert sorted_result[5].get_content() == "chunk2_index1"
    assert sorted_result[5].score == 0.73

    custom_fusion_retriever = il.QueryFusionRetrieverCustom(
        retrievers=[MockRetriever] * 3,
        mode="simple",
    )
    sorted_result = custom_fusion_retriever._simple_fusion(mock_retrieved_result)
    assert len(sorted_result) == 6

    assert sorted_result[0].get_content() == "chunk1_index1"
    assert sorted_result[0].score == 0.75
    assert sorted_result[1].get_content() == "chunk2_index1"
    assert sorted_result[1].score == 0.73
    assert sorted_result[2].get_content() == "chunk1_index2"
    assert round(sorted_result[2].score, 4) == round(
        0.755 * (1 - (1 * 0.05)), 4
    )  # 0.7172
    assert sorted_result[3].get_content() == "chunk1_index3"
    assert round(sorted_result[3].score, 4) == round(
        0.745 * (1 - (1 * 0.05)), 4
    )  # 0.7077
    assert sorted_result[4].get_content() == "chunk2_index3"
    assert round(sorted_result[4].score, 4) == round(
        0.738 * (1 - (1 * 0.05)), 4
    )  # 0.7011
    assert sorted_result[5].get_content() == "chunk2_index2"
    assert round(sorted_result[5].score, 4) == round(
        0.735 * (1 - (1 * 0.05)), 4
    )  # 0.6982
