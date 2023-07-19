"""Utils for the subpackage"""
from collections import Counter
from tqdm import tqdm
import numpy as np
from langdetect import detect
import langid
import fasttext

tqdm.pandas()


class LanguageIdentification:
    """Class to detect the language using fasttext"""

    def __init__(self):
        pretrained_lang_model = r"/dbfs/mnt/datascience/fs-shared/raw/text_translation/identification_models/lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        """Apply the language on a text"""
        predictions = self.model.predict(text, k=1)  # returns top 2 matching languages
        return predictions


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def try_to_detect(text, engine, engine_model=None):
    """
    Try to detect the language in a text.

    Parameters
    ----------
    text: str
        Text to detect the language on
    engine: str
        Technology to use for detection
    engine_model: LanguageIdentification
        Detection language if needed

    Returns
    -------
    lang: str
        Return the language of the text if detected
    """
    engines = ["langdetect", "langid", "fasttext"]
    assert engine in engines + ["ensemble"], f"Engine {engine} not supported"
    try:
        if engine == "langdetect":
            lang = detect(text)
        elif engine == "langid":
            lang = langid.classify(text)[0]
        elif engine == "fasttext":
            lang = engine_model.predict_lang(text)
            lang = lang[0][0][-2:]
        elif engine == "ensemble":
            lang = [
                try_to_detect(text, identification_language, engine_model)
                for identification_language in engines
            ]
            occurence_count = Counter(lang)
            lang = occurence_count.most_common(1)[0][0]
    except:
        lang = np.nan
    return lang


def detect_all_languages(
    df, columns=[], language_prefix="_language", engine="langdetect"
):
    """
    Detect the languages of the provided columns

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing the needed column
    columns: str
        Columns name where the language detection is applied
    language_prefix: str
        Prefix for the column containing the detected language
    engine: str
        Engine to use for language detection

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with new columns containing the detected languages
    """
    if engine in ["fasttext", "ensemble"]:
        print("Using fasttext for identification.")
        language = LanguageIdentification()
    elif engine in ["langid", "langdetect"]:
        language = None
    else:
        raise ValueError(
            "No correct engine specified. Possible engine are: 'langdetect' (default), 'langid' or 'fasttext'"
        )
    for column in columns:
        language_column = column + language_prefix
        df[language_column] = df[column].progress_map(
            lambda text: try_to_detect(text, engine, engine_model=language)
        )
    return df
