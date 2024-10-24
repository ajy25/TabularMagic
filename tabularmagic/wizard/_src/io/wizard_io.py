import qdrant_client
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode
import matplotlib.pyplot as plt
import pathlib


from ..llms.openai import build_openai
from ..llms.vision import describe_image
from .._debug.logger import print_debug


io_path = pathlib.Path(__file__).resolve().parent
img_store_path = io_path / "_img_store"
img_store_path.mkdir(exist_ok=True)


vector_store_path = io_path / "_vector_store"
vector_store_path.mkdir(exist_ok=True)


class WizardIO:

    def __init__(self):
        print_debug("Qdrant client initialized.")
        print_debug("WizardIO initialized.")
        self.reset()

    def add_str(self, text: str) -> str:
        """Store text in the vector index.

        Parameters
        ----------
        text : str
            Text to add to the vector index.

        Returns
        -------
        str
            The input text, verbatim.
        """
        self._index.insert_nodes(nodes=[TextNode(text=text)])
        return text

    def add_figure(
        self,
        fig: plt.Figure,
        text_description: str,
        augment_text_description: bool = True,
    ) -> str:
        """Store a figure as an image in the vector index.

        Parameters
        ----------
        fig : plt.Figure
            Figure to add to the vector index.

        text_description : str
            Description of the figure.

        augment_text_description : bool
            Whether to augment the text description with a vision model,
            by default True

        Returns
        -------
        str
            Description of the figure.
        """
        img_path = img_store_path / f"{self._img_counter}.png"

        fig.savefig(img_path)
        self._img_counter += 1

        if augment_text_description:
            text_description += "\n" + describe_image(img_path, text_description)

        storage_description = "Image Path: " + str(img_path) + "\n\n" + text_description

        self._index.insert_nodes(nodes=[TextNode(text=storage_description)])

        return text_description

    def reset(self):

        # delete all images
        for img in img_store_path.iterdir():
            img.unlink()

        self._img_counter = 0

        client = qdrant_client.QdrantClient(path=vector_store_path)

        if (vector_store_path / "collection" / "text_store").exists():
            client.delete_collection("text_store")

        vector_store = QdrantVectorStore(collection_name="text_store", client=client)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )
        self._index = VectorStoreIndex.from_documents(
            documents=[], storage_context=storage_context
        )

        self._retriever = self._index.as_retriever(similarity_top_k=1)

        self._query_engn = self._index.as_query_engine(llm=build_openai())

    @property
    def retriever(self):
        return self._retriever

    @property
    def query_engine(self):
        return self._query_engn
