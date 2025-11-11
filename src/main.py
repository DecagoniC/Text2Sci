import os
import numpy as np

from extract.text_extractor import DocumentExtractor
from preprocess.chunker import TextPreprocessor
from embedding.embedder import TextEmbedder


text=str(input())
preprocessor=TextPreprocessor(use_lemmatization=True)
print(preprocessor.process(text))