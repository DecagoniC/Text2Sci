import os
import numpy as np
import pickle

from extract.text_extractor import DocumentExtractor
from preprocess.chunker import TextPreprocessor
from embedding.embedder import TextEmbedder
from data_manager.data_manager import DatabaseManager, Chunk
from seeker.seeker import Seeker

seeker=Seeker()
print(seeker.get_raw_answer("С кем поговорила Лопахин в начале пьесы о судьбе сада?")[0])