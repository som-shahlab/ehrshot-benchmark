from datetime import datetime
from typing import List, Optional, Union, Dict, Callable
from abc import ABC, abstractmethod
from femr import Event
from collections import defaultdict

# Define constant time for labeling
CONSTANT_LABEL_TIME = datetime(2024, 1, 1)

# Define headings
EHR_HEADING = "\n\n# Electronic Healthcare Record\n\n"
STATIC_EVENTS_HEADING = "## General Events\n\n"
VISITS_EVENTS_HEADING = "## Medical History\n\n"

def datetime_to_markdown(dt):
    display_date = dt.strftime("%Y-%m-%d %H:%M")
    iso_date = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"[{display_date}]({iso_date})"

def datetimes_to_visit_time(label_dt, visit_dt, duration_days=None):
    days_before_label = (label_dt - visit_dt).days
    if duration_days is not None and duration_days > 0:
        return f"{datetime_to_markdown(visit_dt)} ({days_before_label} days before prediction time, Duration: {duration_days} days)"
    else:
        return f"{datetime_to_markdown(visit_dt)} ({days_before_label} days before prediction time)"

def visit_heading(label_dt: datetime, visit) -> str:
    shifted_visit_dt = CONSTANT_LABEL_TIME - (label_dt - visit.start)
    duration_days = (visit.end - visit.start).days if visit.end is not None else None
    return f"### {visit.description} {datetimes_to_visit_time(CONSTANT_LABEL_TIME, shifted_visit_dt, duration_days)}\n\n"

class SerializationStrategy(ABC):
    @abstractmethod
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        pass

    def get_time_text(self):
        return f"Current time: {datetime_to_markdown(CONSTANT_LABEL_TIME)}\n\n"

    def get_unique_events(self, events: List[Event]) -> List[Event]:
        descriptions = set()
        unique_events = []
        for event in events:
            if event.description not in descriptions:
                descriptions.add(event.description)
                unique_events.append(event)
        return unique_events

    def format_float(self, value: float, decimals: int) -> str:
        formatted = f"{value:.{decimals}f}"
        # Remove trailing zeros after the decimal point
        formatted = formatted.rstrip('0')
        # If all decimal places are zero, remove the decimal point
        if formatted.endswith('.'):
            formatted = formatted[:-1]
        return formatted

    def format_value(self, value: Union[str, int, float]) -> str:
        if isinstance(value, float):
            return self.format_float(value, 2)
        else:
            return str(value)
        
    def serialize_event(self, event: Event, numeric_values=False) -> str:
        """ Create markdown list item of event with value """
        if event.value is None:
            return f"- {event.description}"
        elif isinstance(event.value, (float, int)):
            if numeric_values:
                numeric_str = self.format_value(event.value)
                unit_str = f" [{event.unit}]" if event.unit is not None else ""
                return f"- {event.description}{unit_str}: {numeric_str}"
            else:
                return f"- {event.description}"
        elif isinstance(event.value, str):
            return f"- {event.description}: {event.value}"
        else:
            return f"- {event.description}"

    def serialize_event_list(self, events: List[Event], numeric_values=False, unique_events=False) -> str:
        """ Create markdown list of events with values """
        if unique_events:
            return self.serialize_unique_event_list(events, numeric_values)
        else:
            return '\n'.join([self.serialize_event(event, numeric_values) for event in events])

    def serialize_unique_event_list(self, events: List[Event], numeric_values=False) -> str:
        event_dict = defaultdict(lambda: {'values': [], 'unit': None})
        for event in events:
            event_dict[event.description]['values'].append(event.value)  # type: ignore
            event_dict[event.description]['unit'] = event.unit
        # Remove all None values
        for description, data in event_dict.items():
            if data['values'] is not None:
                data['values'] = [value for value in data['values'] if value is not None]

        serialized_events = []
        for description, data in event_dict.items():
            if numeric_values:
                values_str = ', '.join(map(self.format_value, data['values'] if data['values'] is not None else []))
                unit_str = f" [{data['unit']}]" if data['unit'] is not None else ""
                if values_str:
                    serialized_events.append(f"- {description}{unit_str}: {values_str}")
                else:
                    serialized_events.append(f"- {description}")
            else:
                serialized_events.append(f"- {description}")
            values_str = ', '.join(map(str, data['values'] if data['values'] is not None else []))

        return '\n'.join(serialized_events)

    def list_visits_with_events(self, ehr_serializer, label_time, numeric_values=False, unique_events=False) -> str:
        visit_texts = []
        # Set label time to a constant value for all patients
        for visit in sorted(ehr_serializer.visits, reverse=True):
            visit_text = visit_heading(label_time, visit) + self.serialize_event_list(visit.events, numeric_values=numeric_values, unique_events=unique_events)
            visit_texts.append(visit_text)

        return '\n\n'.join(visit_texts)
    
class ListUniqueEventsWoNumericValuesStrategy(SerializationStrategy):
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        events = ehr_serializer.static_events + [event for visit in ehr_serializer.visits for event in visit.events]
        events = sorted(events, key=lambda x: x.start)
        unique_events = self.get_unique_events(events)
        return EHR_HEADING + STATIC_EVENTS_HEADING + self.serialize_event_list(unique_events, numeric_values=False)
    
class ListVisitsWithUniqueEventsWoNumericValuesStrategy(SerializationStrategy):   
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        static_text = STATIC_EVENTS_HEADING + self.serialize_unique_event_list(ehr_serializer.static_events, numeric_values=False)
        visits_text = VISITS_EVENTS_HEADING + self.list_visits_with_events(ehr_serializer, label_time, numeric_values=False, unique_events=True)
        return EHR_HEADING + self.get_time_text() + f"{static_text}\n\n{visits_text}"

class ListVisitsWithUniqueEventsStrategy(SerializationStrategy):
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        static_text = STATIC_EVENTS_HEADING + self.serialize_unique_event_list(ehr_serializer.static_events, numeric_values=True)
        visits_text = VISITS_EVENTS_HEADING + self.list_visits_with_events(ehr_serializer, label_time, numeric_values=True, unique_events=True)
        return EHR_HEADING + self.get_time_text() + f"{static_text}\n\n{visits_text}"

class ListVisitsWithEventsWoNumericValuesStrategy(SerializationStrategy):   
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        static_text = STATIC_EVENTS_HEADING + self.serialize_event_list(ehr_serializer.static_events, numeric_values=False, unique_events=False)
        visits_text = VISITS_EVENTS_HEADING + self.list_visits_with_events(ehr_serializer, label_time, numeric_values=False, unique_events=False)
        return EHR_HEADING + self.get_time_text() + f"{static_text}\n\n{visits_text}"

class ListVisitsWithEventsStrategy(SerializationStrategy):
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        static_text = STATIC_EVENTS_HEADING + self.serialize_event_list(ehr_serializer.static_events, numeric_values=True, unique_events=False)
        visits_text = VISITS_EVENTS_HEADING + self.list_visits_with_events(ehr_serializer, label_time, numeric_values=True, unique_events=False)
        return EHR_HEADING + self.get_time_text() + f"{static_text}\n\n{visits_text}"

class EHRVisit:
    def __init__(
        self,
        visit_id: int,
        start: datetime,
        end: Optional[datetime] = None,
        description: Optional[str] = ""
    ):
        self.visit_id: int = visit_id
        self.start: datetime = start
        self.end: Optional[datetime] = end
        self.description: Optional[str] = description
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
        description: Optional[str] = "",
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
        
    def load_from_femr_events(self, events: List[Event], resolve_code: Callable[[str], Optional[str]], is_visit_event: Callable[[Event], bool]) -> None:

        # First process all visits
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
                    description=description,
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