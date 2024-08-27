from datetime import datetime
from typing import List, Optional, Union, Dict, Callable
from abc import ABC, abstractmethod
from femr import Event
import re
        
class SerializationStrategy(ABC):
    @abstractmethod
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        pass
    
def serialize_event(event: Event, numeric_values=False) -> str:
    """ Create dashed line of event with value """
    if type(event.value) is float or type(event.value) is int:
        if numeric_values:
            numeric_str = f"{event.value:.2f}" if type(event.value) is float else f"{event.value}"
            return f"- {event.description}: {numeric_str}"
        else:
            return f"- {event.description}"
    elif type(event.value) is str:
        return f"- {event.description}: {event.value}"
    else:
        return f"- {event.description}"
    
def serialize_event_list(events: List[Event], numeric_values=False) -> str:
    """ Create dashed list of events with values """
    return '\n'.join([serialize_event(event, numeric_values) for event in events])

def list_visits_with_events(ehr_serializer, label_time, numeric_values=False):
    # Implement the logic to list all visits and their respective events
    visit_texts = []
    for visit in sorted(ehr_serializer.visits):
        days_before_label = (label_time - visit.start).days
        visit_text = f"{days_before_label} days before: {visit.description}\n{serialize_event_list(visit.events, numeric_values=numeric_values)}"
        visit_texts.append(visit_text)
        
    return '\n\n'.join(visit_texts)
    
class ListUniqueEventsWoNumericValuesStrategy(SerializationStrategy):
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        descriptions = set()
        
        events = ehr_serializer.static_events + [event for visit in ehr_serializer.visits for event in visit.events]
        events = sorted(events, key=lambda x: x.start)
        unique_events = []
        for event in events:
            if event.description not in descriptions:
                descriptions.add(event.description)
                unique_events.append(event)
                
        return serialize_event_list(unique_events, numeric_values=False)
    
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

class ListVisitsWithEventsWoNumericValuesStrategy(SerializationStrategy):   
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        static_text = serialize_event_list(ehr_serializer.static_events, numeric_values=False)
        visits_text = list_visits_with_events(ehr_serializer, label_time, numeric_values=False)
        return f"{static_text}\n\n{visits_text}"

class ListVisitsWithEventsStrategy(SerializationStrategy):
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        static_text = serialize_event_list(ehr_serializer.static_events, numeric_values=True)
        visits_text = list_visits_with_events(ehr_serializer, label_time, numeric_values=True)
        return f"{static_text}\n\n{visits_text}"

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

    def serialize(self, serialization_strategy: SerializationStrategy, label_time: datetime) -> str:
        return serialization_strategy.serialize(self, label_time)