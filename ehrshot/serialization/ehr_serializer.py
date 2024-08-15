from datetime import datetime
from typing import List, Optional, Union, Dict, Callable
from abc import ABC, abstractmethod
from femr import Event
import re
        
class SerializationStrategy(ABC):
    @abstractmethod
    def serialize(self, ehr_serializer):
        pass
    
class ListUniqueEventsStrategy(SerializationStrategy):
    def serialize(self, ehr_serializer):
        text_events = []
        text_set = set()
        
        events = ehr_serializer.static_events + [event for visit in ehr_serializer.visits for event in visit.events]
        events = sorted(events, key=lambda x: x.start)
        for event in events:

            # TODO: Potential processing of decriptions
            # # Remove some suffixes:
            # # 'in Serum or Plasma', 'Serum or Plasma', ' - Serum or Plasma', 'in Serum', 'in Plasma'
            # # 'in Blood', ' - Blood', 'in Blood by Automated count', 'by Automated count', ', automated'
            # # 'by Manual count'
            # re_exclude_description_suffixes = re.compile(r"( in Serum or Plasma| Serum or Plasma| - Serum or Plasma| in Serum| in Plasma| in Blood| - Blood| in Blood by Automated count| by Automated count|, automated| by Manual count)")
            
            # # Remove some irrelevant artifacts
            # # Remove all [*] - often correspond to units
            # description = re.sub(r"\[.*\]", "", description)
            # # Remove suffixes
            # description = re_exclude_description_suffixes.sub("", description)
            # # Remove repeated whitespaces
            # description = re.sub(r"\s+", " ", description)
            # description = description.strip()
            
            # Each code only once
            if event.description in text_set:
                continue
            text_set.add(event.description)
            
            if type(event.value) is str:
                text_events.append(f"- {event.description}: {event.value}")
            # TODO: Add handling of numeric values
            else:
                text_events.append(f"- {event.description}")
            
        text = '\n'.join(text_events)
        return text

class ListVisitsWithEventsStrategy(SerializationStrategy):
    def serialize(self, ehr_serializer):
        # Implement the logic to list all visits and their respective events
        return '\n'.join([f"{visit.description}: {', '.join([event.description for event in visit.events])}" for visit in ehr_serializer.visits])

class EHRVisit:
    def __init__(
        self,
        visit_id: int,
        start: datetime,
        end: Optional[datetime] = None,
        description: str = "",
    ):
        self.visit_id: int = visit_id
        self.start: datetime = start
        self.end: Optional[datetime] = end
        self.description: str = description
        self.events: List[EHREvent] = []

    def add_event(self, event: 'EHREvent') -> None:
        self.events.append(event)

    def __lt__(self, other: 'EHRVisit') -> bool:
        if self.start != other.start:
            return self.start < other.start
        if self.end is None and other.end is None:
            return False
        if self.end is None:
            return False
        if other.end is None:
            return True
        return self.end < other.end

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EHRVisit):
            return NotImplemented
        return (self.visit_id == other.visit_id and
                self.start == other.start and
                self.end == other.end and
                self.description == other.description)

class EHREvent:
    def __init__(
        self,
        start: datetime,
        end: Optional[datetime] = None,
        description: str = "",
        value: Optional[Union[str, int, float]] = None,
        unit: Optional[str] = None,
        serialization_rank: float = 0.0,
    ):
        self.start = start
        self.end = end
        self.description = description
        self.value = value
        self.unit = unit

class EHRSerializer:
    def __init__(self):
        self.visits: List[EHRVisit] = []
        self.static_events: List[EHREvent] = []
        
    def load_from_femr_events(self, events: List[Event], resolve_code: Callable[[str], str], is_visit_event: Callable[[Event], bool]) -> None:

        # First process all visit
        visit_ids_to_visits: Dict[int, EHRVisit] = {}
        for event_visit in filter(is_visit_event, events):
            description = resolve_code(event_visit.code)
            if description is not None:
                visit = EHRVisit(
                    visit_id=event_visit.visit_id,
                    start=event_visit.start,
                    end=event_visit.end if hasattr(event_visit, 'end') else None,
                    description=description
                )
                visit_ids_to_visits[event_visit.visit_id] = visit
        
        # Then process all events
        for event in filter(lambda x: not is_visit_event(x), events):
            visit = visit_ids_to_visits.get(event.visit_id, None)
            description = resolve_code(event.code)
            if description is not None:
                event = EHREvent(
                    start=event.start,
                    end=event.end if hasattr(event, 'end') else None,
                    description=resolve_code(event.code),
                    value=event.value if hasattr(event, 'value') else None,
                    unit=event.unit if hasattr(event, 'unit') else None
                )
                if visit is not None:
                    visit.add_event(event)
                else:
                    self.static_events.append(event)
                    
        self.visits = sorted(visit_ids_to_visits.values())
        
    def set_serialization_strategy(self, serialization_strategy: SerializationStrategy):
        self._serialization_strategy = serialization_strategy

    def serialize(self, serialization_strategy: SerializationStrategy):
        return serialization_strategy.serialize(self)