from seeker.seeker import Seeker
from data_manager.data_manager import DatabaseManager
import os

seeker=Seeker()
print(seeker.get_raw_answer("Какая сцена показывает прощание с садом в рассказе Чехова Вишневый сад, и кто в ней участвует?",2))