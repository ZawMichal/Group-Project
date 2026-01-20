import cv2
import shutil
import os
import csv
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
        self._blacklist_cache = {}
        self._blacklist_dirty = True
        self._ensure_blacklist_file()
        self._sync_blacklist_with_people()

    def _blacklist_path(self):
        return config.blacklist_csv_path

    def _ensure_blacklist_file(self):
        path = self._blacklist_path()
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "blacklisted"])

    def _load_blacklist(self):
        self._ensure_blacklist_file()
        data = {}
        try:
            with open(self._blacklist_path(), "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("name", "").strip()
                    value = row.get("blacklisted", "0").strip()
                    if name:
                        data[name] = value == "1"
        except Exception:
            pass
        return data

    def _save_blacklist(self, data):
        with open(self._blacklist_path(), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "blacklisted"])
            for name in sorted(data.keys()):
                writer.writerow([name, "1" if data[name] else "0"])
        self._blacklist_cache = data
        self._blacklist_dirty = False

    def _sync_blacklist_with_people(self):
        people = set(self.get_people_list())
        data = self._load_blacklist()

        # Add missing people
        for name in people:
            if name not in data:
                data[name] = False

        # Remove entries for deleted people
        for name in list(data.keys()):
            if name not in people:
                del data[name]

        self._save_blacklist(data)
        return data

    def _get_blacklist_cache(self):
        if self._blacklist_dirty or not self._blacklist_cache:
            self._sync_blacklist_with_people()
        return self._blacklist_cache

    def get_people_list(self):
        """Zwraca listę nazw osób (nazw folderów)."""
        if not self.known_faces_dir.exists():
            return []
        return sorted([d.name for d in self.known_faces_dir.iterdir() if d.is_dir()])

    def get_blacklist_status(self, name):
        data = self._get_blacklist_cache()
        return bool(data.get(name, False))

    def set_blacklist_status(self, name, is_blacklisted: bool):
        data = self._get_blacklist_cache()
        if name:
            data[name] = bool(is_blacklisted)
            self._save_blacklist(data)
            return True
        return False

    def create_person_folder(self, name):
        """Tworzy folder dla nowej osoby. Bezpieczne dla nazw ze spacjami."""
        clean_name = "".join([c for c in name if c.isalnum() or c in (' ', '_', '-')]).strip()
        if not clean_name: return None
        
        person_dir = self.known_faces_dir / clean_name
        person_dir.mkdir(exist_ok=True)
        # Ensure person exists in blacklist file
        self.set_blacklist_status(clean_name, False)
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
                # Remove from blacklist file
                data = self._get_blacklist_cache()
                if name in data:
                    del data[name]
                    self._save_blacklist(data)
                return True
            except Exception:
                return False
        return False