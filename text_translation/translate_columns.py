import numpy as np

from text_translation.utils import detect_all_languages
from text_translation.translate_dataframe import TranslateDataframe


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
    Translate a dataframe column according to a translation dict. Allows to translate a column containing many languages.

    :param translation_dict: dictionarry containing the translation mapping
    :param df: df to work on
    :param column: column name to translate
    :param language_prefix: prefix for the language column
    :param translation_prefix: prefix for the translated column
    :param single_target: precise if the target is always the same. If it is the case, the translated columns will be filled by raw text already in the target language.
    :type translation_dict: dict
    :type df: pd.DataFrame
    :type column: str
    :type language_prefix: str
    :type translation_prefix: str
    :type single_target: bool
    :return: dataframe with the translated columns
    :rtype: pd.DataFrame
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
