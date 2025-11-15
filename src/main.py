from seeker.seeker import Seeker
from data_manager.data_manager import DatabaseManager
import os


# db = DatabaseManager(data_path="data")

# folder_path = "D:/Go-prog/prog2/articles"  

# for filename in os.listdir(folder_path):
#     filepath = os.path.join(folder_path, filename)
    
#     if os.path.isfile(filepath):
#         try:
#             print(f"[+] Добавляется файл: {filename}")
#             db.add_article(filepath)
#         except Exception as e:
#             print(f"[!] Ошибка при добавлении {filename}: {e}")

# print("[✓] Все файлы добавлены в базу.")




seeker=Seeker()
print(seeker.get_raw_answer("Какая сцена показывает прощание с садом в рассказе Чехова Вишневый сад, и кто в ней участвует?",5))