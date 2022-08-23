from text_translation.translation_model import TranslationModel


def test_translate_cpu():
    fr_2_en_cpu = TranslationModel("fr", "en", device=-1)

    translated_text = fr_2_en_cpu.translate(
        ["Bonjour, je m'apelle erwan", "Comment tu vas ?"]
    )
    assert isinstance(translated_text, list)
    assert translated_text == ["Hello, I call myself erwan.", "How are you?"]


def test_translate_gpu():
    fr_2_en_gpu = TranslationModel("fr", "en", device=0)
    translated_text = fr_2_en_gpu.translate(
        ["Bonjour, je m'apelle erwan", "Comment tu vas ?"]
    )
    assert isinstance(translated_text, list)
    assert translated_text == ["Hello, I call myself erwan.", "How are you?"]
