import qdrant_client
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import TextNode, ImageNode
import matplotlib.pyplot as plt
import pathlib


def print_debug(str):
    print(str)
def describe_image(img_path):
    return "A beautiful image."
from dotenv import load_dotenv
import pathlib
import os
from typing import Literal


def find_key(llm_type: Literal["openai"]) -> str:
    """Reads the .env file and returns the API key for the specified LLM type.
    If the API key is not found, raises a ValueError.

    Parameters
    ----------
    llm_type : Literal["openai"]
        The type of LLM for which to find the API key.
    """
    load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent.parent / ".env")

    if llm_type == "openai":
        api_key = (
            str(os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        )
        if api_key == "..." or api_key is None:
            raise ValueError("OpenAI API key not found in .env file.")
    else:
        raise ValueError("Invalid LLM type specified.")

    return api_key

os.environ["OPENAI_API_KEY"] = find_key("openai")

# from ..llms.openai import build_openai
# from ..llms.vision import describe_image
# from .._debug.logger import print_debug


io_path = pathlib.Path(__file__).resolve().parent
img_store_path = io_path / "_img_store"
img_store_path.mkdir(exist_ok=True)
qdrant_store_path = io_path / "_qdrant_store"
qdrant_store_path.mkdir(exist_ok=True)
qdrant_path = qdrant_store_path / "qdrant_img_db"

client = qdrant_client.QdrantClient(
    path=qdrant_path
)
text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)
print("Creating index...")
index = MultiModalVectorStoreIndex.from_documents(
    documents=[],
    storage_context=storage_context,
    # embed_model=OpenAIEmbedding(
    #     api_key=find_key("openai")
    # )
)


class WizardIO:

    def __init__(self):
        print_debug("Qdrant client initialized.")
        print_debug("WizardIO initialized.")

    def add_str(self, text: str):
        """Store text in the vector index.

        Parameters
        ----------
        text : str
            Text to add to the vector index.
        """
        index.insert_nodes(nodes=[TextNode(text=text)])
        

    def add_figure(
        self, 
        fig: plt.Figure, 
        text_description: str,
        augment_text_description: bool = False
    ):
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

        if augment_text_description:
            text_description += "\n" + describe_image(img_path)

        index.insert_nodes(
            nodes=[ImageNode(image_path=img_path, text=text_description)]
        )


    def reset(self):

        # delete all images
        for img in img_store_path.iterdir():
            img.unlink()

        self._img_counter = 0

        self._retriever = index.as_retriever(
            similarity_top_k=1, 
            image_similarity_top_k=1
        )

        # self._query_engn = self._index.as_query_engine(
        #     llm=build_openai()
        # )

    @property
    def retriever(self):
        return self._retriever
    
    # @property
    # def query_engine(self):
    #     return self._query_engn


GLOBAL_IO = WizardIO()
GLOBAL_IO.reset()

print("Adding text to the index...")

GLOBAL_IO.add_str("The meaning of life is exactly 42.")

print(GLOBAL_IO.retriever.retrieve("What is the meaning of life?"))

