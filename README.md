# Introduction 
This projet aim to provide an easy way to translate texts locally without using expternal API or tool which could break the confidentiality of the data. 
It can be used for small text translation or directly to translate columns of dataframe.

The library use MarianMT architecture which is a translation model based on the work of Microsoft. Several models has been trained to cover most of the languages. However, it may be possible that the language you want to translate is not available.

Full list of available models can be found here:
https://huggingface.co/Helsinki-NLP
The first translation from a language to another can take time as the tool need to download the right model for the translation task.

Note: GPU need to be enabled for the library to work properly. If you are using it on your local computer, you should probably install Cuda:
https://developer.nvidia.com/cuda-10.1-download-archive-base

# Getting Started
The translation_example notebook gives an example on how to use the library.

### Translate a columns from a dataframe having multiples languages
```python
from text_translation.translate_columns import translate_multiple_languages
import pandas as pd
translation_dict = {
    "fr2en": {
        "languages": ["fr"],
        "model_source": "fr",
        "model_target": "en"
    },
    "es2en": {
        "languages": ["es"],
        "model_source": "es",
        "model_target": "en"
    }
}
df = pd.DataFrame(["Hello, I don't know how to speak French", 
                    "Ce n'est pas grave je peux parler anglais si besoin.", 
                    "yes, that would be very nice of you", 
                    "De rien, cela me fait plaisir", "Hola, ¿cómo estás?", 
                    "No entiendo el inglés, lo siento."], columns=["raw_texts"]
                    )
translated_df = translate_multiple_languages(translation_dict, df, "raw_texts", single_target=True, gpu=False)
```

### Translate a list of text from French to English
```python
from source.translation_model import TranslationModel

fr_2_en = TranslationModel("fr", "en")
fr_2_en.translate(["Bonjour, je m'apelle Erwan", "Comment tu vas ?"])
```

# Build and Test
Run 'python -m pytest' to launch the tests

# Contribute
TODO list:
- Allowing the library to work on CPU.
- Improve documentation