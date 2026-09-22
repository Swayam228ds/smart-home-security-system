import json
import logging
import threading
import os
from typing import Optional, List, Union
from pydantic import BaseModel

from server.config import settings

logger = logging.getLogger(__name__)

class Owner(BaseModel):
    name: str
    status: str  # 'home', 'away', 'busy', 'sleeping'
    notify_on_visitor: bool

class OwnerManager:
    """Manages home owners' status and notifications."""
    def __init__(self, file_path: str = None):
        self.file_path = file_path or settings.appointments_file
        self.lock = threading.RLock()
        self.owners: List[Owner] = []
        self._last_mtime = 0.0
        self.reload()

    def reload(self):
        """Reloads owners from JSON file if it has been modified."""
        with self.lock:
            try:
                if not os.path.exists(self.file_path):
                    logger.warning(f"Appointments file not found at {self.file_path}")
                    return

                mtime = os.path.getmtime(self.file_path)
                if mtime > self._last_mtime:
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    self.owners = [Owner(**owner_data) for owner_data in data.get("owners", [])]
                    self._last_mtime = mtime
                    logger.info("Owners loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading owners: {e}")

    def _save(self):
        """Private method to write the owners configuration back to JSON."""
        with self.lock:
            try:
                # Read the whole JSON structure to update only the owners part
                if os.path.exists(self.file_path):
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = {}
                
                # Update owners list via Pydantic v2 model_dump
                data["owners"] = [owner.model_dump() for owner in self.owners]
                
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                
                self._last_mtime = os.path.getmtime(self.file_path)
                logger.info("Owners saved successfully.")
            except Exception as e:
                logger.error(f"Error saving owners: {e}")

    def get_owner_status(self, name: str = None) -> Union[Owner, List[Owner], None]:
        """Gets the status for a specific owner by name, or all owners if name is None."""
        self.reload()
        with self.lock:
            if name:
                for owner in self.owners:
                    if owner.name.lower() == name.lower():
                        return owner
                return None
            return list(self.owners)

    def get_all_owners(self) -> List[Owner]:
        """Returns all owners."""
        self.reload()
        with self.lock:
            return list(self.owners)

    def update_status(self, name: str, status: str) -> Optional[Owner]:
        """Updates the status of an owner and saves to JSON."""
        self.reload()
        with self.lock:
            for owner in self.owners:
                if owner.name.lower() == name.lower():
                    owner.status = status
                    self._save()
                    return owner
            return None

    def is_anyone_home(self) -> bool:
        """Checks if any owner currently has status 'home'."""
        self.reload()
        with self.lock:
            return any(owner.status == 'home' for owner in self.owners)

    def get_available_owners(self) -> List[Owner]:
        """Returns a list of owners with status 'home'."""
        self.reload()
        with self.lock:
            return [owner for owner in self.owners if owner.status == 'home']

    def notify_owner(self, owner_name: str, visitor_name: str, purpose: str) -> None:
        """Logs a notification for the specified owner."""
        # This is a placeholder for a future push notification mechanism
        logger.info(f"NOTIFICATION to {owner_name}: Visitor '{visitor_name}' is here for '{purpose}'.")
