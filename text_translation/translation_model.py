import gc
import os
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, pipeline, Pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset


class TranslationModel:
    """
    TranslationModel is a class to load a translation model and apply it on a list of texts.

    Arguments:
        source: Source language of the texts that need translation
        target: Target language for the translation

    Usage:

    ```python
    fr_to_en = TranslationModel('fr', 'en')
    fr_texts = [">>en<< Je vous présente TranslationModel", ">>en<< C'est plutot sympa"]
    en_texts = translate(fr_texts)
    ```
    """

    def __init__(self, source, target, device=-1):
        self.source: str = source
        self.target: str = target
        self.models_path: str = "/dbfs/mnt/dsl/fs-shared/raw/text_translation/translation_models/"
        self.model_name: str = f"Helsinki-NLP/opus-mt-{source}-{target}"
        self.model_path: str = os.path.join(self.models_path, self.model_name)
        self.tokenizer: MarianTokenizer = None
        self.model: MarianMTModel = None
        self.translater: Pipeline = None
        self.device: int = device
        self._load_model()

    def _load_model(self):
        try:
            print("Try loading from custom folder")
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_path)
            self.model = MarianMTModel.from_pretrained(self.model_path)
        except:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name)
            print("Model not found in custom folder, loading from HuggingFace Hub")
        self.translater = pipeline(
            task="translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            truncation=True
        )
        print(f"Model to translate from {self.source} to {self.target} loaded.")

    def translate(self, text_list):
        """
        Translate the given text to the target language

        :param text_list: list of text to translate
        :type text_list: list
        :return: return list of translated text
        :rtype: list
        """
        df = Dataset.from_dict({"to_process": text_list})
        translated_text = [out[0]["translation_text"] for out in tqdm(self.translater(KeyDataset(df, "to_process"), batch_size=8), total=len(df))]

        # translated_text = self.translater(text_list)
        # translated_text = [item["translation_text"] for item in translated_text]
        return translated_text

    def clean_cuda(self):
        """
        Clean the cuda environment to release memory.
        """
        if "self.model" in vars() or "self.model" in globals():
            del self.model
        if "self.tokenizer" in vars() or "self.tokenizer" in globals():
            del self.tokenizer
        gc.collect()
        # torch.cuda.empty_cache()
        return True
