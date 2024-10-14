from pathlib import Path
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageDocument

from .openai import build_openai_multimodal


general_purpose_vision_model: OpenAIMultiModal = build_openai_multimodal()


def describe_image(image_path: Path, text_description: str) -> str:
    """Describes an image using the multimodal model.

    Parameters
    ----------
    image_path : Path
        Path to the image.

    text_description : str
        Non AI generated supplementary text description to add to the image description.

    Returns
    -------
    str
        Description of the image.
    """
    image_document = ImageDocument(image_path=str(image_path))

    prompt = "Describe the figure in detail. Be specific. "
    "Here is some additional information: "
    f"{text_description}"

    str_description = general_purpose_vision_model.complete(
        prompt=prompt, image_documents=[image_document]
    ).text
    return str_description
