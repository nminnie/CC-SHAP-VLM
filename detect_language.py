from lingua import Language, LanguageDetectorBuilder

def load_lang_detector():
    languages = [
        Language.BENGALI, Language.GERMAN,
        Language.ENGLISH, Language.INDONESIAN,
        Language.KOREAN, Language.PORTUGUESE,
        Language.RUSSIAN, Language.CHINESE,
    ]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    return detector


def detect_lang(detector, text):
    if text.strip().lower() == "no":
        return "en"
    try:
        language = detector.detect_language_of(text)
        lang_code = language.iso_code_639_1.name.lower()
    except Exception:
        lang_code = ""

    return lang_code
