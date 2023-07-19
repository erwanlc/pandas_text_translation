"""Script to do multiple translation on multiple languages"""
import numpy as np

from dsl.text_translation.utils import detect_all_languages
from dsl.text_translation.translate_dataframe import TranslateDataframe


def translate_multiple_languages(
    translation_dict,
    df,
    column,
    language_prefix="_language",
    translation_prefix="_translated",
    single_target=True,
    gpu=False,
    engine="langdetect",
):
    """
    Translate a dataframe column according to a translation dict.
    Allows to translate a column containing many languages.

    Parameters
    ----------
    translation_dict: dict
        Dictionarry containing the translation mapping
    df: pandas.DataFrame
        Df to work on
    column: str
        Column name to translate
    language_prefix: str
        Prefix for the language column
    translation_prefix: str
        Prefix for the translated column
    single_target: bool
        Precise if the target is always the same. If it is the case,
        the translated columns will be filled by raw text already in the target language.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with the translated columns
    """
    print("Detecting language")
    df = detect_all_languages(df, [column], engine=engine)
    df[column + translation_prefix] = np.nan
    if single_target:
        df[column + translation_prefix] = df[column + translation_prefix].fillna(
            df[
                df[column + language_prefix]
                == list(translation_dict.items())[0][1]["model_target"]
            ][column]
        )
    for translation in translation_dict:
        print(translation)
        df_lang = df[
            df[column + language_prefix].isin(
                translation_dict[translation]["languages"]
            )
        ]
        if len(df_lang) == 0:
            print("No text to translate for that language")
            continue
        df_lang = TranslateDataframe(
            df_lang,
            column,
            translation_dict[translation]["model_source"],
            translation_dict[translation]["model_target"],
            gpu=gpu,
        )
        df_lang.translate_column()
        df[column + translation_prefix] = df[column + translation_prefix].fillna(
            df_lang.get_translated_column()
        )
    return df
