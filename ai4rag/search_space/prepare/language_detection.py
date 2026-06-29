# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import re

from ai4rag import logger
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel

LANGUAGE_MAP: dict[str, str] = {
    "aa": "Afar",
    "ab": "Abkhazian",
    "af": "Afrikaans",
    "ak": "Akan",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "av": "Avaric",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bh": "Bihari",
    "bi": "Bislama",
    "bm": "Bambara",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "ce": "Chechen",
    "ch": "Chamorro",
    "co": "Corsican",
    "cr": "Cree",
    "cs": "Czech",
    "cu": "Church Slavic",
    "cv": "Chuvash",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dv": "Divehi",
    "dz": "Dzongkha",
    "ee": "Ewe",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "ff": "Fulah",
    "fi": "Finnish",
    "fj": "Fijian",
    "fo": "Faroese",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "gl": "Galician",
    "gn": "Guarani",
    "gu": "Gujarati",
    "gv": "Manx",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "ho": "Hiri Motu",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "hz": "Herero",
    "ia": "Interlingua",
    "id": "Indonesian",
    "ie": "Interlingue",
    "ig": "Igbo",
    "ii": "Sichuan Yi",
    "ik": "Inupiaq",
    "io": "Ido",
    "is": "Icelandic",
    "it": "Italian",
    "iu": "Inuktitut",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kg": "Kongo",
    "ki": "Kikuyu",
    "kj": "Kuanyama",
    "kk": "Kazakh",
    "kl": "Kalaallisut",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "kr": "Kanuri",
    "ks": "Kashmiri",
    "ku": "Kurdish",
    "kv": "Komi",
    "kw": "Cornish",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lg": "Ganda",
    "li": "Limburgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lu": "Luba-Katanga",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mh": "Marshallese",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "na": "Nauru",
    "nb": "Norwegian Bokmal",
    "nd": "North Ndebele",
    "ne": "Nepali",
    "ng": "Ndonga",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "nr": "South Ndebele",
    "nv": "Navajo",
    "ny": "Chichewa",
    "oc": "Occitan",
    "oj": "Ojibwe",
    "om": "Oromo",
    "or": "Odia",
    "os": "Ossetian",
    "pa": "Punjabi",
    "pi": "Pali",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "rm": "Romansh",
    "rn": "Rundi",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "sa": "Sanskrit",
    "sc": "Sardinian",
    "sd": "Sindhi",
    "se": "Northern Sami",
    "sg": "Sango",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sm": "Samoan",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "ss": "Swati",
    "st": "Southern Sotho",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "ti": "Tigrinya",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tn": "Tswana",
    "to": "Tonga",
    "tr": "Turkish",
    "ts": "Tsonga",
    "tt": "Tatar",
    "tw": "Twi",
    "ty": "Tahitian",
    "ug": "Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "ve": "Venda",
    "vi": "Vietnamese",
    "vo": "Volapuk",
    "wa": "Walloon",
    "wo": "Wolof",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "za": "Zhuang",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zu": "Zulu",
}


def detect_language_with_llm(
    questions: list[str],
    generation_model: OGXFoundationModel,
) -> dict[str, str] | None:
    """Detect the dominant language from sample questions using an LLM.

    Sends a small sample of questions to a generation model registered in OGX
    and asks it to return the ISO 639-1 code.  Models listed in
    *allowed_generation_models* are preferred when available.

    Parameters
    ----------
    questions : list[str]
        Raw question texts to classify.  Only the first five are sent to the model.

    generation_model : OGXFoundationModel
        Model instance.

    Returns
    -------
    dict[str, str] | None
        A dictionary with ``code`` and ``name`` keys when a non-English
        language is detected, or ``None`` for English / on failure.
    """
    sample_text = "\n".join(f"- {q}" for q in questions[:5])

    try:
        response = generation_model.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a language detection assistant. "
                        "Given text samples, respond with ONLY the ISO 639-1 language code. "
                        "Nothing else — just the code."
                    ),
                },
                {
                    "role": "user",
                    "content": f"What language are these questions written in?\n{sample_text}",
                },
            ],
            max_completion_tokens=10,
            temperature=0.0,
        )
        raw_content = response[0].message.content
        if not raw_content or not isinstance(raw_content, str):
            raise ValueError(f"Invalid response content: {type(raw_content)}")

        cleaned = raw_content.strip().lower().replace('"', "").replace("'", "")
        if not cleaned:
            raise ValueError("Empty response after cleanup")

        code_pattern = r"[a-z]{2}(?:-[a-z]{2,4})?"
        # Try targeted patterns to avoid matching English stop words (e.g. "is", "it", "no")
        match = (
            re.match(rf"^({code_pattern})\s*$", cleaned)  # code only
            or re.match(rf"^({code_pattern})\s", cleaned)  # code at start, then more text
            or re.search(rf"\(({code_pattern})\)", cleaned)  # code in parentheses
        )
        if not match:
            raise ValueError(f"No ISO 639-1 code found in response: {cleaned[:50]}")

        detected_code = match.group(1).split("-")[0]
        name = LANGUAGE_MAP.get(match.group(1)) or LANGUAGE_MAP.get(detected_code)
        if not name:
            raise ValueError(f"Unsupported language code '{detected_code}' from response: {cleaned[:50]}")

        logger.info("Language detected via LLM: %s (%s)", detected_code, name)
        return {"code": detected_code, "name": name}

    except Exception as exc:
        logger.warning("LLM language detection failed: %s", exc)
        return None
