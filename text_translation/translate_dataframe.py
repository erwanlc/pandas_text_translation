"""Class to translate a dataframe"""
import numpy as np

from dsl.text_translation.translation_model import TranslationModel


class TranslateDataframe:
    """
    TranslateDataframe is a class to translate a dataframe column from source to target language.
    During the translation, two additional columns are created,
    one for the text prepared and one for the translated text.

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
    df_fr = pd.DataFrame(
        [">>en<< Je vous présente TranslationModel",
        ">>en<< C'est plutot sympa"], columns=["raw_texts"]
        )

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
        gpu=False,
    ):
        self.df = df
        self.column = column
        self.model_source = model_source
        self.model_target = model_target
        self.prepared_column = column + prepare_prefix
        self.translation_column = column + translation_prefix
        self.chunk_size = chunk_size
        self.device = -1
        if gpu:
            self.device = 0

    def get_translated_column(self):
        """
        Get the translated column

        Returns
        -------
        Translated column: pandas.Series
            Translated column
        """
        return self.df[self.translation_column]

    def prepare_for_translation(self):
        """
        Prepare the column for translation by adding target language at the beginning of the text.

        Returns
        -------
        df: pandas.DataFrame
            Dataframe with prepared column
        """
        self.df[self.prepared_column] = self.df[self.column].map(
            lambda text: self.try_to_prepare(text)
        )
        return self.df

    def translate_column(self):
        """
        Translate the needed column

        Returns
        -------
        translated_texts: list
            list of translated texts
        """
        translation_model = TranslationModel(
            self.model_source, self.model_target, self.device
        )
        print("Number of texts to translate: ", len(self.df))
        prepared_texts = self.df[self.column].tolist()

        translated_texts = translation_model.translate(prepared_texts)
        self.df[self.translation_column] = translated_texts
        return translated_texts

    def try_to_prepare(self, text):
        """
        Try to prepare for translation a given text

        Parameters
        ----------
        text: str
            Text to prepare

        Returns
        -------
        prepared_line: str
            Prepared text
        """
        try:
            prepared_line = f">>{self.model_target}<< {text}"
        except:
            prepared_line = np.nan
        return prepared_line
