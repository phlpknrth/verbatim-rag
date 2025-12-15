"""
Cloud Milvus storage implementation with BM25 full-text search support.
"""

import logging
from typing import List, Optional

from .base import SearchResult
from .milvus_base import BaseMilvusStore
from .hybrid_search import convert_hits_to_results

logger = logging.getLogger(__name__)


class CloudMilvusStore(BaseMilvusStore):
    """
    Cloud Milvus storage with hybrid search and BM25 full-text search support.

    Supports Milvus Standalone, Distributed, and Zilliz Cloud.
    """

    def __init__(
        self,
        collection_name: str = "verbatim_rag",
        dense_dim: int = 384,
        sparse_dim: int = 30000,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        enable_dense: bool = True,
        enable_sparse: bool = True,
        enable_full_text: bool = True,
        index_type: str = "IVF_FLAT",
        nlist: int = 8192,
    ):
        if not uri:
            raise ValueError("CloudMilvusStore requires 'uri' for cloud connections")

        self.uri = uri
        self.token = token
        self.sparse_dim = sparse_dim
        self._needs_dummy_vector = True  # Cloud requires vectors in all collections

        super().__init__(
            collection_name=collection_name,
            dense_dim=dense_dim,
            enable_dense=enable_dense,
            enable_sparse=enable_sparse,
            enable_full_text=enable_full_text,
            index_type=index_type,
            nlist=nlist,
        )

        self._setup_client()

    def _get_dynamic_fields(self) -> List[str]:
        """Return dynamic field names for Cloud Milvus queries."""
        return ["user_id", "document_id", "dataset_id"]

    def _setup_client(self):
        """Initialize Cloud Milvus client and create collections."""
        try:
            from pymilvus import MilvusClient, DataType

            if self.token:
                self.client = MilvusClient(uri=self.uri, token=self.token)
            else:
                self.client = MilvusClient(uri=self.uri)

            # Create main chunks collection if missing
            if not self.client.has_collection(collection_name=self.collection_name):
                schema = self.client.create_schema(
                    auto_id=False, enable_dynamic_field=True
                )

                schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=512,
                )

                if self.enable_dense:
                    schema.add_field(
                        field_name="dense_vector",
                        datatype=DataType.FLOAT_VECTOR,
                        dim=self.dense_dim,
                    )

                if self.enable_sparse:
                    schema.add_field(
                        field_name="sparse_vector",
                        datatype=DataType.SPARSE_FLOAT_VECTOR,
                    )

                schema.add_field(
                    field_name="text",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                )
                schema.add_field(
                    field_name="enhanced_text",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                )
                schema.add_field(field_name="metadata", datatype=DataType.JSON)

                # Add BM25 function for full text search if enabled
                if self.enable_full_text:
                    self._add_bm25_to_schema(schema, DataType)

                self.client.create_collection(
                    collection_name=self.collection_name, schema=schema
                )

                # Create indexes
                index_params = self.client.prepare_index_params()

                if self.enable_dense:
                    index_params.add_index(
                        field_name="dense_vector",
                        index_type=self.index_type,
                        metric_type="COSINE",
                        params={"nlist": self.nlist},
                    )

                if self.enable_sparse:
                    index_params.add_index(
                        field_name="sparse_vector",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP",
                        params={"inverted_index_algo": "DAAT_MAXSCORE"},
                    )

                if self.enable_full_text:
                    self._add_bm25_index(index_params)

                self.client.create_index(
                    collection_name=self.collection_name, index_params=index_params
                )

                # Load collection into memory
                try:
                    self.client.load_collection(collection_name=self.collection_name)
                    logger.info(f"Loaded collection: {self.collection_name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to load collection {self.collection_name}: {e}"
                    )

                logger.info(f"Created cloud Milvus collection: {self.collection_name}")

            # Create documents collection
            if not self.client.has_collection(
                collection_name=self.documents_collection_name
            ):
                self._create_documents_collection()

            logger.info("Connected to Milvus Cloud via MilvusClient")

        except ImportError:
            raise ImportError("pip install pymilvus")

    def _add_bm25_to_schema(self, schema, DataType):
        """Add BM25 function to schema for full text search."""
        try:
            from pymilvus import Function, FunctionType

            schema.add_field(
                field_name="bm25_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )

            bm25_function = Function(
                name="text_bm25_emb",
                input_field_names=["text"],
                output_field_names=["bm25_vector"],
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)

        except ImportError:
            logger.warning(
                "BM25 function not available. Full text search will be disabled. "
                "Ensure you have a compatible version of pymilvus."
            )
            self.enable_full_text = False

    def _add_bm25_index(self, index_params):
        """Add BM25 index for full text search."""
        try:
            index_params.add_index(
                field_name="bm25_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "bm25_k1": 1.2,
                    "bm25_b": 0.75,
                },
            )
        except Exception as e:
            logger.warning(
                f"Failed to create BM25 index: {e}. Full text search may not work."
            )
            self.enable_full_text = False

    def _create_documents_collection(self):
        """Create documents collection with dummy vector for cloud constraint."""
        from pymilvus import DataType

        doc_schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)

        doc_schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=100,
        )
        doc_schema.add_field(
            field_name="title", datatype=DataType.VARCHAR, max_length=4096
        )
        doc_schema.add_field(
            field_name="source", datatype=DataType.VARCHAR, max_length=4096
        )
        doc_schema.add_field(
            field_name="content_type", datatype=DataType.VARCHAR, max_length=50
        )
        doc_schema.add_field(
            field_name="raw_content", datatype=DataType.VARCHAR, max_length=65535
        )
        doc_schema.add_field(field_name="metadata", datatype=DataType.JSON)
        # Cloud Milvus requires a vector field in all collections
        doc_schema.add_field(
            field_name="dummy_vector", datatype=DataType.FLOAT_VECTOR, dim=2
        )

        self.client.create_collection(
            collection_name=self.documents_collection_name, schema=doc_schema
        )

        # Create flat index on dummy_vector so collection can be loaded
        doc_index_params = self.client.prepare_index_params()
        doc_index_params.add_index(
            field_name="dummy_vector",
            index_type="FLAT",
            metric_type="L2",
            params={},
        )
        self.client.create_index(
            collection_name=self.documents_collection_name,
            index_params=doc_index_params,
        )

        # Load the documents collection into memory
        try:
            self.client.load_collection(collection_name=self.documents_collection_name)
            logger.info(
                f"Loaded documents collection: {self.documents_collection_name}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load documents collection {self.documents_collection_name}: {e}"
            )

        logger.info(
            f"Created cloud Milvus documents collection: {self.documents_collection_name}"
        )

    def _full_text_search(
        self, text_query: str, top_k: int, filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform full text search using BM25.

        Args:
            text_query: Natural language text query
            top_k: Number of results to return
            filter: Optional filter expression

        Returns:
            List of SearchResult objects
        """
        if not self.enable_full_text:
            raise ValueError("Full text search is not enabled for this collection")

        try:
            dynamic_fields = self._get_dynamic_fields()
            output_fields = self._get_output_fields()

            results = self.client.search(
                collection_name=self.collection_name,
                data=[text_query],
                anns_field="bm25_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
            )

            return convert_hits_to_results(results[0], dynamic_fields)

        except Exception as e:
            logger.error(f"Full text search failed: {e}")
            return []
