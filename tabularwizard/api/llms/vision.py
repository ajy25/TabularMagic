from pathlib import Path
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageDocument

from .openai import build_openai_multimodal


general_purpose_vision_model: OpenAIMultiModal = build_openai_multimodal()

def describe_image(image_path: Path) -> str:
    """Describes an image using the multimodal model.

    Parameters
    ----------
    image_path : Path
        Path to the image.

    Returns
    -------
    str
        Description of the image.
    """
    image_document = ImageDocument(image_path=str(image_path))
    str_description = general_purpose_vision_model.complete(
        prompt="Describe the image.",
        image_documents=[image_document]
    ).text
    return str_description

