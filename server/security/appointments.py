import json
import logging
import threading
import os
import difflib
from datetime import date, time, datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel

from server.config import settings

logger = logging.getLogger(__name__)

class Appointment(BaseModel):
    visitor_name: str
    purpose: str
    date: str
    time_range: str
    approved: bool

class DefaultRules(BaseModel):
    allow_deliveries: bool
    allow_known_visitors: bool
    deny_unknown_after_hours: bool
    quiet_hours: str

class AppointmentsConfig(BaseModel):
    owners: list
    appointments: List[Appointment]
    default_rules: DefaultRules

class AppointmentManager:
    """Manages home security appointments and rules loading/saving."""
    def __init__(self, file_path: str = None):
        self.file_path = file_path or settings.appointments_file
        self.lock = threading.RLock()
        self.config: Optional[AppointmentsConfig] = None
        self._last_mtime = 0.0
        self.reload()

    def reload(self):
        """Reloads appointments from JSON file if it has been modified."""
        with self.lock:
            try:
                if not os.path.exists(self.file_path):
                    logger.warning(f"Appointments file not found at {self.file_path}")
                    return

                mtime = os.path.getmtime(self.file_path)
                if mtime > self._last_mtime:
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.config = AppointmentsConfig(**data)
                    self._last_mtime = mtime
                    logger.info("Appointments loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading appointments: {e}")

    def _save(self):
        """Private method to write the current configuration back to JSON."""
        with self.lock:
            if not self.config:
                return
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    # model_dump is available in Pydantic v2
                    data = self.config.model_dump()
                    json.dump(data, f, indent=2)
                self._last_mtime = os.path.getmtime(self.file_path)
                logger.info("Appointments saved successfully.")
            except Exception as e:
                logger.error(f"Error saving appointments: {e}")

    def get_all_appointments(self) -> List[Appointment]:
        """Returns a list of all appointments."""
        self.reload()
        with self.lock:
            if not self.config:
                return []
            return self.config.appointments

    def find_appointment(self, name: str, purpose: str, current_date: date = None) -> Optional[Appointment]:
        """
        Find an appointment based on fuzzy name matching, purpose keywords, date, and time range.
        Returns the best matching appointment or None.
        """
        self.reload()
        if current_date is None:
            current_date = date.today()
        
        current_time = datetime.now().time()
        current_date_str = current_date.isoformat()
        
        best_match = None
        best_score = 0.0
        
        with self.lock:
            if not self.config:
                return None
            
            for appt in self.config.appointments:
                # Check date validity
                if appt.date != current_date_str and appt.date.lower() not in ("all_day", "today", "any"):
                    continue
                
                # Check time range validity
                is_valid_time = False
                if appt.time_range.lower() == "all_day":
                    is_valid_time = True
                else:
                    try:
                        start_str, end_str = appt.time_range.split('-')
                        start_t = datetime.strptime(start_str.strip(), "%H:%M").time()
                        end_t = datetime.strptime(end_str.strip(), "%H:%M").time()
                        if start_t <= end_t:
                            is_valid_time = start_t <= current_time <= end_t
                        else:
                            is_valid_time = current_time >= start_t or current_time <= end_t
                    except ValueError:
                        # Fallback if format is unusual
                        is_valid_time = True
                
                if not is_valid_time:
                    continue
                
                # Fuzzy name matching
                matcher = difflib.SequenceMatcher(None, name.lower(), appt.visitor_name.lower())
                score = matcher.ratio()
                
                # Purpose keyword matching
                if purpose:
                    purpose_match = any(word.lower() in appt.purpose.lower() for word in purpose.split())
                    if purpose_match:
                        score += 0.2
                    
                if score >= 0.6 and score > best_score:
                    best_score = score
                    best_match = appt
                    
        return best_match

    def add_appointment(self, appointment: Appointment) -> None:
        """Adds a new appointment and saves to JSON."""
        self.reload()
        with self.lock:
            if self.config:
                self.config.appointments.append(appointment)
                self._save()

    def remove_appointment(self, visitor_name: str, date_str: str) -> bool:
        """Removes an appointment by visitor name and date, saving to JSON."""
        self.reload()
        with self.lock:
            if not self.config:
                return False
            initial_len = len(self.config.appointments)
            self.config.appointments = [
                appt for appt in self.config.appointments 
                if not (appt.visitor_name.lower() == visitor_name.lower() and appt.date == date_str)
            ]
            if len(self.config.appointments) < initial_len:
                self._save()
                return True
            return False

    def is_delivery(self, purpose: str) -> bool:
        """Checks if the purpose string matches typical delivery keywords."""
        if not purpose:
            return False
        delivery_keywords = ['delivery', 'package', 'amazon', 'fedex', 'ups', 'mail', 'courier']
        return any(keyword in purpose.lower() for keyword in delivery_keywords)

    def check_rules(self, purpose: str, current_time: time = None) -> dict:
        """Evaluates default_rules regarding deliveries, known visitors, and quiet hours."""
        self.reload()
        if current_time is None:
            current_time = datetime.now().time()
            
        with self.lock:
            if not self.config:
                return {}
            
            rules = self.config.default_rules
            
            # Evaluate quiet hours
            quiet = False
            if rules.quiet_hours:
                try:
                    start_str, end_str = rules.quiet_hours.split('-')
                    start_t = datetime.strptime(start_str.strip(), "%H:%M").time()
                    end_t = datetime.strptime(end_str.strip(), "%H:%M").time()
                    if start_t <= end_t:
                        quiet = start_t <= current_time <= end_t
                    else:
                        quiet = current_time >= start_t or current_time <= end_t
                except ValueError:
                    logger.error("Invalid quiet_hours format. Expected HH:MM-HH:MM")
            
            return {
                "allow_deliveries": rules.allow_deliveries,
                "allow_known_visitors": rules.allow_known_visitors,
                "deny_unknown_after_hours": rules.deny_unknown_after_hours,
                "is_quiet_hours": quiet
            }
