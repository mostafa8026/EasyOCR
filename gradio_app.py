import pandas as pd
from PIL import Image, ImageDraw
import gradio as gr
import torch
import easyocr
from pathlib import Path

SAMPLES_DIR = Path("demo_samples")
SAMPLE_IMAGES = {
    "english.png": "https://github.com/JaidedAI/EasyOCR/raw/master/examples/english.png",
    "thai.jpg": "https://github.com/JaidedAI/EasyOCR/raw/master/examples/thai.jpg",
    "french.jpg": "https://github.com/JaidedAI/EasyOCR/raw/master/examples/french.jpg",
    "chinese.jpg": "https://github.com/JaidedAI/EasyOCR/raw/master/examples/chinese.jpg",
    "japanese.jpg": "https://github.com/JaidedAI/EasyOCR/raw/master/examples/japanese.jpg",
    "korean.png": "https://github.com/JaidedAI/EasyOCR/raw/master/examples/korean.png",
    "Hindi.jpeg": "https://i.imgur.com/mwQFd7G.jpeg",
}

CHOICES = [
    "abq",
    "ady",
    "af",
    "ang",
    "ar",
    "as",
    "ava",
    "az",
    "be",
    "bg",
    "bh",
    "bho",
    "bn",
    "bs",
    "ch_sim",
    "ch_tra",
    "che",
    "cs",
    "cy",
    "da",
    "dar",
    "de",
    "en",
    "es",
    "et",
    "fa",
    "fr",
    "ga",
    "gom",
    "hi",
    "hr",
    "hu",
    "id",
    "inh",
    "is",
    "it",
    "ja",
    "kbd",
    "kn",
    "ko",
    "ku",
    "la",
    "lbe",
    "lez",
    "lt",
    "lv",
    "mah",
    "mai",
    "mi",
    "mn",
    "mr",
    "ms",
    "mt",
    "ne",
    "new",
    "nl",
    "no",
    "oc",
    "pi",
    "pl",
    "pt",
    "ro",
    "ru",
    "rs_cyrillic",
    "rs_latin",
    "sck",
    "sk",
    "sl",
    "sq",
    "sv",
    "sw",
    "ta",
    "tab",
    "te",
    "th",
    "tjk",
    "tl",
    "tr",
    "ug",
    "uk",
    "ur",
    "uz",
    "vi",
]

CSS = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"


def ensure_samples():
    """Download sample images into the demo folder if they are missing."""
    SAMPLES_DIR.mkdir(exist_ok=True)
    for filename, url in SAMPLE_IMAGES.items():
        target_path = SAMPLES_DIR / filename
        if not target_path.exists():
            torch.hub.download_url_to_file(url, str(target_path))


def sample_path(filename: str) -> str:
    """Return the str path for a sample image inside the demo folder."""
    return str(SAMPLES_DIR / filename)


def draw_boxes(image, bounds, color="yellow", width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image


def inference(img_path, lang):
    ensure_samples()
    reader = easyocr.Reader(lang, gpu=torch.cuda.is_available())
    bounds = reader.readtext(img_path)
    image = Image.open(img_path)
    draw_boxes(image, bounds).save("result.jpg")
    df = pd.DataFrame(bounds)
    if df.empty:
        df = pd.DataFrame(columns=["text", "confidence"])
    else:
        df = df.iloc[:, 1:]
        df.columns = ["text", "confidence"]
    return "result.jpg", df


examples = [
    [sample_path("english.png"), ["en"]],
    [sample_path("thai.jpg"), ["th"]],
    [sample_path("french.jpg"), ["fr", "en"]],
    [sample_path("chinese.jpg"), ["ch_sim", "en"]],
    [sample_path("japanese.jpg"), ["ja", "en"]],
    [sample_path("korean.png"), ["ko", "en"]],
    [sample_path("Hindi.jpeg"), ["hi", "en"]],
]

title = "EasyOCR"
description = (
    "Gradio demo for EasyOCR. Upload your image and choose a language from the dropdown "
    "menu, or click one of the examples to load them."
)
article = (
    "<p style='text-align: center'><a href='https://www.jaided.ai/easyocr/'>Ready-to-use OCR with "
    "80+ supported languages and all popular writing scripts including Latin, Chinese, Arabic, "
    "Devanagari, Cyrillic, etc.</a> | <a href='https://github.com/JaidedAI/EasyOCR'>Github Repo</a></p>"
)

ensure_samples()

gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(type="filepath", label="Input"),
        gr.CheckboxGroup(choices=CHOICES, value=["en"], label="language"),
    ],
    outputs=[
        gr.Image(type="filepath", label="Output"),
        gr.Dataframe(headers=["text", "confidence"]),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
    css=CSS,
).launch()
