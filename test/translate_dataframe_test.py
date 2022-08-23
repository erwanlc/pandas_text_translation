import pytest
import numpy as np
import pandas as pd
from text_translation.translate_dataframe import TranslateDataframe

TEXT_LIST = ["Bonjour, je m'apelle erwan", "Comment tu vas ?"]

DF_TEST = pd.DataFrame(TEXT_LIST, columns=["raw_texts"])

DF_LANG = TranslateDataframe(DF_TEST, "raw_texts", "fr", "en")


def test_try_to_prepare():
    prepared_text = DF_LANG.try_to_prepare("Ceci est du français")
    assert prepared_text == ">>en<< Ceci est du français"


def test_prepare_for_translation():
    result = DF_LANG.prepare_for_translation()
    assert isinstance(result, pd.DataFrame)
    assert "raw_texts_prepared" in result.columns


def test_translate_column_cpu():
    translated_texts = DF_LANG.translate_column()
    assert isinstance(translated_texts, list)
    assert translated_texts == ["Hello, I call myself erwan.", "How are you?"]

def test_translate_column_gpu():
    df_lang = TranslateDataframe(DF_TEST, "raw_texts", "fr", "en", gpu=True)
    translated_texts = df_lang.translate_column()
    assert isinstance(translated_texts, list)
    assert translated_texts == ["Hello, I call myself erwan.", "How are you?"]
