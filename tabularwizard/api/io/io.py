import qdrant_client
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode, ImageNode, IndexNode
import matplotlib.pyplot as plt
import pathlib

from ..llms.find_key_from_dot_env import find_key


io_path = pathlib.Path(__file__).resolve().parent
img_store_path = io_path / "_img_store"
img_store_path.mkdir(exist_ok=True)
qdrant_store_path = io_path / "_qdrant_store"
qdrant_store_path.mkdir(exist_ok=True)


class WizardIO:

    def __init__(self):
        qdrant_path = qdrant_store_path / "qdrant_img_db"

        self._client = qdrant_client.QdrantClient(path=qdrant_path)

    def add_str(self, text: str):
        """Store text in the vector index.

        Parameters
        ----------
        text : str
            Text to add to the vector index.
        """
        self._index.insert_nodes(nodes=[TextNode(text=text)])

    def add_figure(self, fig: plt.Figure, text_description: str):
        """Store a figure as an image in the vector index.

        Parameters
        ----------
        fig : plt.Figure
            Figure to add to the vector index.

        text_description : str
            Description of the figure.
        """
        img_path = img_store_path / f"{self._img_counter}.png"

        fig.savefig(img_path)
        self._img_counter += 1

        self._index.insert_nodes(
            nodes=[ImageNode(image_path=img_path, text=text_description)]
        )

    def add_query_engine(self, query_engine, text_description: str):
        self._index.insert_nodes(
            nodes=[IndexNode(obj=query_engine, text=text_description)]
        )

    def reset(self):
        # clear contents of the vector index
        self._client.delete_collection(collection_name="text_collection")
        self._client.delete_collection(collection_name="image_collection")

        # local vector store
        text_store = QdrantVectorStore(
            client=self._client, collection_name="text_collection"
        )
        image_store = QdrantVectorStore(
            client=self._client, collection_name="image_collection"
        )
        index_store = QdrantVectorStore(
            client=self._client, collection_name="index_collection"
        )
        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store, index_store=index_store
        )

        self._index = MultiModalVectorStoreIndex(
            storage_context=storage_context,
            embed_model=OpenAIEmbedding(api_key=find_key("openai")),
        )

        # delete all images
        for img in img_store_path.iterdir():
            img.unlink()

        self._img_counter = 0

    def as_retriever(self):
        return self._index.as_retriever(similarity_top_k=1, image_similarity_top_k=1)


GLOBAL_IO = WizardIO()
