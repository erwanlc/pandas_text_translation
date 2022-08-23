import pytest
import numpy as np
import pandas as pd
from text_translation import utils

TEXT_LIST = [
    "Hello, I don't know how to speak French",
    "Ce n'est pas grave je peux parler anglais si besoin.",
    "yes, that would be very nice of you",
    "De rien, cela me fait plaisir",
    "Hola, ¿cómo estás?",
    "No entiendo el inglés, lo siento.",
]

DF_TEST = pd.DataFrame(TEXT_LIST, columns=["raw_texts"])

ENGINE = "langdetect"

@pytest.fixture()
def list_to_chunk():
    chunk_size = 10
    list_to_test = list(range(0, 30, 1))
    return list_to_test, chunk_size


def test_chunks(list_to_chunk):
    list_to_test, chunk_size = list_to_chunk
    chunked_list = utils.chunks(list_to_test, chunk_size)
    assert len(list(chunked_list)) == 3


def test_try_to_detect():
    text = TEXT_LIST[0]
    language = utils.try_to_detect(text, "langdetect")
    assert language == "en"
    text = 4
    language = utils.try_to_detect(text, "langdetect")
    assert np.isnan(language)

    text = TEXT_LIST[0]
    language = utils.try_to_detect(text, "langid")
    assert language == "en"
    text = 4
    language = utils.try_to_detect(text, "langid")
    assert np.isnan(language)


def test_detect_all_languages():
    df_detected = utils.detect_all_languages(DF_TEST, DF_TEST.columns)
    assert isinstance(df_detected, pd.DataFrame)
