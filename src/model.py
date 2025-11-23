from transformers import AutoModelForTokenClassification
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model
