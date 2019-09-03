# -*- coding: utf-8 -*-
###############################################################################
# MODULE: parsing
# this is a repo of tools designed to parse any filetype
#
# Assumptions: Ubuntu environment for textract
# Author:   William Kinsman
# Created:  10.10.2015
"""
need to add:
    extract text from url
"""
###############################################################################
import os
import re

# import textract
import urllib

# import html2text
# import speech_recognition as sr
from text_tools import preprocessing


def build_document_from_dir(directory=None):
    """
	@param directory : a path to a dir. Defaults to cwd.
	return: list of strings, where a string is from each txt file
	"""
    doc = []
    for f in os.listdir(directory):
        if f[-4:] == ".txt":
            doc.append(open(f, "r").read())
    return doc


def extract_text(fileORurl):

    # fetch if url
    if is_url(fileORurl):
        try:
            with urllib.request.urlopen(fileORurl) as response:
                text = response.read().decode("utf-8")
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h = h.handle(text)

            return h.handle(text)
        except:
            raise Exception("The site could not be parsed. Aborting.")
            return None

    # get file if file
    assert os.path.isfile(fileORurl)
    try:
        text = textract.process(fileORurl)
        assert (
            text != None or text != ""
        ), "Text could not be parsed. Aborting."
    except:
        raise Exception("The file could not be parsed. Aborting.")
    return text


def is_url(url):
    """
    @param url: a string in the form of a url
    """
    re_url = re.compile(
        r"((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:"\
            r"www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w_-]*)?\??"\
                r"(?:[-\+=&;%@.\w_]*)#?(?:[\w]*))?)"
    )
    url = url.strip()
    if re_url.match(url):
        return True
    else:
        return False


def check_connection():
    try:
        urllib.request.urlopen("http://216.58.192.142", timeout=1)
        return True
    except urllib.request.URLError:
        return False


def pdf_extract(filepath):
    import pytesseract
    from tqdm import tqdm
    from pdf2image import convert_from_path

    pytesseract.pytesseract.tesseract_cmd = (
        r"""C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"""
    )

    text = []
    for i in tqdm(convert_from_path(filepath)):
        text.append(pytesseract.image_to_string(i))
    return text


def audio_extract(filepath):
    """
    given a wav file pull the text
    """
    # rip the text from the audio file
    r = sr.Recognizer()
    framerate = 100
    with sr.AudioFile(filepath) as source:
        audio = r.record(source)
        decoder = r.recognize_sphinx(audio, show_all=True)
        stamps = [
            (
                preprocessing.strip_parenthesized(seg.word),
                seg.start_frame / framerate,
            )
            for seg in decoder.seg()
        ]
        stamps = stamps[1:-1]
    return stamps
