from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

class ConversationState(Enum):
    IN_PROGRESS = auto()
    GREETING = auto() # Legacy compatibility if needed
    APPROVED = auto()
    DENIED = auto()
    ENDED = auto()

@dataclass
class ConversationContext:
    session_id: str
    state: ConversationState = ConversationState.IN_PROGRESS
    visitor_name: Optional[str] = None
    visitor_purpose: Optional[str] = None
    matched_appointment: Optional[Any] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
