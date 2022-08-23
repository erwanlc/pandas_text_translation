from itertools import chain
import pandas as pd
import numpy as np
from tqdm import tqdm

from text_translation.translation_model import TranslationModel
from text_translation.utils import chunks


class TranslateDataframe:
    """
    TranslateDataframe is a class to translate a dataframe column of a source language into the target language.
    During the translation, two additional columns are created, one for the text prepared and one for the translated text.

    Arguments:
        df: dataframe to work on
        column: column name to translate
        model_source: language of the column to translate
        model_target: language of the translation
        prepare_prefix: prefix for the data preparation column
        translation_prefix: prefix for the translated column
        chunk_size: number of texts to send to the model at once

    Usage:

    ```python
    df_fr = pd.DataFrame([">>en<< Je vous présente TranslationModel", ">>en<< C'est plutot sympa"], columns=["raw_texts"])

    translate_df = TranslateDataframe(
            df_fr,
            "raw_texts",
            "fr",
            "en",
    )
    translate_df.translate_column()
    print(translate_df.get_translated_df())
    ```
    """

    def __init__(
        self,
        df,
        column,
        model_source,
        model_target,
        prepare_prefix="_prepared",
        translation_prefix="_translation",
        chunk_size=20,
        gpu=False
    ):
        self.df = df
        self.column = column
        self.model_source = model_source
        self.model_target = model_target
        self.prepared_column = column + prepare_prefix
        self.translation_column = column + translation_prefix
        self.chunk_size = chunk_size
        self.device=-1
        if gpu:
            self.device=0
        

    def get_translated_column(self):
        """
        Get the translated column
        :return: Translated column
        :rtype: pd.Series
        """
        return self.df[self.translation_column]

    def prepare_for_translation(self):
        """
        Prepare the column for translation by adding the target language at the beginning of the text
        :return: Dataframe with prepared column
        :rtype: pd.DataFrame
        """
        self.df[self.prepared_column] = self.df[self.column].map(
            lambda text: self.try_to_prepare(text)
        )
        return self.df

    def translate_column(self):
        """
        Translate the needed column
        :return: list of translated texts
        :rtype: list
        """
        translation_model = TranslationModel(self.model_source, self.model_target, self.device)
        #self.prepare_for_translation()
        print("Number of texts to translate: ", len(self.df))
        prepared_texts = self.df[self.column].tolist()
        
        #prepared_texts = list(chunks(prepared_texts, self.chunk_size))
        # translated_texts = []
        # for text_chunk in tqdm(prepared_texts):
        #     translated_texts.append(translation_model.translate(text_chunk))
        # translated_texts = list(chain.from_iterable(translated_texts))
        translated_texts = translation_model.translate(prepared_texts)
        self.df[self.translation_column] = translated_texts
        return translated_texts

    def try_to_prepare(self, text):
        """
        Try to prepare for translation a given text
        """
        try:
            prepared_line = f">>{self.model_target}<< {text}"
        except:
            prepared_line = np.nan
        return prepared_line
