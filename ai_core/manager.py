import cv2
import shutil
import os
from pathlib import Path
from .config import config

class PersonManager:
    """
    Klasa zarządzająca plikami i folderami osób (Baza Danych Twarzy).
    """
    def __init__(self):
        self.known_faces_dir = config.known_faces_dir
        # Automatyczne tworzenie folderu jeśli nie istnieje (wymóg pkt 4)
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)

    def get_people_list(self):
        """Zwraca listę nazw osób (nazw folderów)."""
        if not self.known_faces_dir.exists():
            return []
        return sorted([d.name for d in self.known_faces_dir.iterdir() if d.is_dir()])

    def create_person_folder(self, name):
        """Tworzy folder dla nowej osoby. Bezpieczne dla nazw ze spacjami."""
        clean_name = "".join([c for c in name if c.isalnum() or c in (' ', '_', '-')]).strip()
        if not clean_name: return None
        
        person_dir = self.known_faces_dir / clean_name
        person_dir.mkdir(exist_ok=True)
        return person_dir

    def rename_person(self, old_name, new_name):
        """Zmienia nazwę folderu osoby (Edycja)."""
        old_dir = self.known_faces_dir / old_name
        
        clean_new_name = "".join([c for c in new_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        if not clean_new_name: return False
        
        new_dir = self.known_faces_dir / clean_new_name
        
        if old_dir.exists() and not new_dir.exists():
            try:
                # shutil.move zmienia nazwę katalogu
                shutil.move(str(old_dir), str(new_dir))
                return True
            except Exception as e:
                print(f"[MANAGER BŁĄD] Nie udało się zmienić nazwy: {e}")
                return False
        return False

    def save_training_photo(self, name, frame):
        """Zapisuje klatkę do folderu danej osoby."""
        person_dir = self.create_person_folder(name)
        if not person_dir: return None
        
        existing = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        count = len(existing) + 1
        
        filename = person_dir / f"{name}_{count}.jpg"
        try:
            cv2.imwrite(str(filename), frame)
            return str(filename)
        except Exception as e:
            print(f"[MANAGER BŁĄD] Zapis pliku nieudany: {e}")
            return None

    def delete_person(self, name):
        """Usuwa osobę (cały folder)."""
        person_dir = self.known_faces_dir / name
        if person_dir.exists():
            try:
                shutil.rmtree(person_dir)
                return True
            except Exception:
                return False
        return False