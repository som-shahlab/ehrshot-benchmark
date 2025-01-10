from datetime import datetime
from typing import List, Optional, Union, Dict, Callable, Tuple
from abc import ABC, abstractmethod
from femr import Event
from collections import defaultdict
import re

# Define constant time for labeling
CONSTANT_LABEL_TIME = datetime(2024, 1, 1)

# Define headings
EHR_HEADING = "\n\n# Electronic Healthcare Record\n\n"
STATIC_EVENTS_HEADING = "## General Medical Events\n\n"
VISITS_EVENTS_HEADING = "## Medical History\n\n"
MEDICATIONS_EVENTS_HEADING = "## Recent Medications\n\n"

AGGREGATED_SUB_EVENTS = {
    'Body Metrics': {
        'heading': "## Recent Body Metrics\n",
        'events': ['Body weight', 'Body height', 'Body mass index / BMI', 'Body surface area']
    },
    'Vital Signs': {
        'heading': "## Recent Vital Signs\n",
        'events': ['Heart rate', 'Respiratory rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Body temperature', 'Oxygen saturation']
    },
    'Lab Results': {
        'heading': "## Recent Lab Results\n",
        'events': ['Hemoglobin', 'Hematocrit', 'Erythrocytes', 'Leukocytes', 'Platelets', 'Sodium', 'Potassium', 'Chloride', 'Carbon dioxide, total', 'Calcium', 'Glucose', 'Urea nitrogen', 'Creatinine', 'Anion gap']
    }
}

def format_int(x):
    return f"{int(x)}"

def format_one_decimal(x):
    return f"{x:.1f}"

def format_two_decimals(x):
    return f"{x:.2f}"

AGGREGATED_EVENTS = {
    'Heart rate': {
        'codes': ['LOINC/8867-4', 'SNOMED/364075005', 'SNOMED/78564009'],
        'min_max': [5, 300],
        'normal_range': [60, 100],
        'unit': 'bpm',
        'format': lambda x: f"{int(x)}"
    },
    'Systolic blood pressure': {
        'codes': ['LOINC/8480-6', 'SNOMED/271649006'],
        'min_max': [20, 300],
        'normal_range': [90, 140], # European Society of Cardiology
        'unit': 'mmHg',
        'format': lambda x: f"{int(x)}"
    },
    'Diastolic blood pressure': {
        'codes': ['LOINC/8462-4', 'SNOMED/271650006'],
        'min_max': [20, 300],
        'normal_range': [60, 90],  # European Society of Cardiology
        'unit': 'mmHg',
        'format': lambda x: f"{int(x)}"
    },
    'Body temperature': {
        'codes': ['LOINC/8310-5'], 
        'min_max': [80, 120],
        'normal_range': [95, 100,4],  # 35 - 38 °C
        'unit': '°F',
        'format': format_one_decimal
    },
    'Respiratory rate': {
        'codes': ['LOINC/9279-1'],
        'min_max': [1, 100],
        'normal_range': [12, 18],
        'unit': 'breaths/min',
        'format': lambda x: f"{int(x)}"
    },
    'Oxygen saturation': {
        'codes': ['LOINC/LP21258-6'],
        'min_max': [1, 100],
        'normal_range': [95, 100],
        'unit': '%',
        'format': lambda x: f"{int(x)}"
    },
    
    'Body weight': {
        'codes': ['LOINC/29463-7'],
        'min_max': [350, 10000],
        'unit': 'oz',
        'format': format_one_decimal
    },
    'Body height': {
        'codes': ['LOINC/8302-2'],
        'min_max': [5, 100],
        'unit': 'inch',
        'format': format_one_decimal
    },
    'Body mass index / BMI': {
        'codes': ['LOINC/39156-5'],
        'min_max': [10, 100],
        'normal_range': [18.5, 24.9],
        'unit': 'kg/m2',
        'format': format_one_decimal
    },
    'Body surface area': {
        'codes': ['LOINC/8277-6', 'SNOMED/301898006'],
        'min_max': [0.1, 10],
        'unit': 'm2',
        'format': format_two_decimals
    },

    # Normal values: https://annualmeeting.acponline.org/sites/default/files/shared/documents/for-meeting-attendees/normal-lab-values.pdf
    'Hemoglobin': {
        'codes': ['LOINC/718-7', 'SNOMED/271026005', 'SNOMED/441689006'],
        'min_max': [1, 20],
        'normal_range': [12, 17], # combining female and male
        'unit': 'g/dL',
        'format': format_one_decimal
    },
    'Hematocrit': {
        'codes': ['LOINC/4544-3', 'LOINC/20570-8', 'LOINC/48703-3', 'SNOMED/28317006'],
        'min_max': [10, 100],
        'normal_range': [36, 51], # combining female and male
        'unit': '%',
        'format': lambda x: f"{int(x)}"
    },
    'Erythrocytes': {
        'codes': ['LOINC/789-8', 'LOINC/26453-1'],
        'min_max': [1, 10],
        'normal_range': [4.2, 5.9],
        'unit': '10^6/uL',
        'format': format_two_decimals
    },
    'Leukocytes': {
        'codes': ['LOINC/20584-9', 'LOINC/6690-2'],
        'min_max': [1, 100],
        'normal_range': [4, 10],
        'unit': '10^3/uL',
        'format': format_one_decimal
    },
    'Platelets': {
        'codes': ['LOINC/777-3', 'SNOMED/61928009'],
        'min_max': [10, 1000],
        'normal_range': [150, 350],
        'unit': '10^3/uL',
        'format': lambda x: f"{int(x)}"
    },
    
    'Sodium': {
        'codes': ['LOINC/2951-2', 'LOINC/2947-0', 'SNOMED/25197003'],
        'min_max': [100, 200],
        'normal_range': [136, 145],
        'unit': 'mmol/L',
        'format': lambda x: f"{int(x)}"
    },
    'Potassium': {
        'codes': ['LOINC/2823-3', 'SNOMED/312468003', 'LOINC/6298-4', 'SNOMED/59573005'],
        'min_max': [0.1, 10],
        'normal_range': [3.5, 5.0],
        'unit': 'mmol/L',
        'format': format_one_decimal
    },
    'Chloride': {
        'codes': ['LOINC/2075-0', 'SNOMED/104589004', 'LOINC/2069-3'],
        'min_max': [50, 200],
        'normal_range': [98, 106],
        'unit': 'mmol/L',
        'format': lambda x: f"{int(x)}"
    },
    'Carbon dioxide, total': {
        'codes': ['LOINC/2028-9'],
        'min_max': [10, 100],
        'normal_range': [23, 28],
        'unit': 'mmol/L',
        'format': lambda x: f"{int(x)}"
    },
    'Calcium': {
        'codes': ['LOINC/17861-6', 'SNOMED/271240001'],
        'min_max': [1, 20],
        'normal_range': [9, 10.5],
        'unit': 'mg/dL',
        'format': format_one_decimal
    },
    'Glucose': {
        'codes': ['LOINC/2345-7', 'SNOMED/166900001', 'LOINC/2339-0', 'SNOMED/33747003', 'LOINC/14749-6'],
        'min_max': [10, 1000],
        'normal_range': [70, 100],
        'unit': 'mg/dL',
        'format': lambda x: f"{int(x)}"
    },
    'Urea nitrogen': {
        'codes': ['LOINC/3094-0', 'SNOMED/105011006'],
        'min_max': [1, 200],
        'normal_range': [8, 20],
        'unit': 'mg/dL',
        'format': lambda x: f"{int(x)}"
    },
    'Creatinine': {
        'codes': ['LOINC/2160-0', 'SNOMED/113075003'],
        'min_max': [0.1, 10],
        'normal_range': [0.7, 1.3],
        'unit': 'mg/dL',
        'format': format_one_decimal
    },
    'Anion gap': {
        'codes': ['LOINC/33037-3', 'LOINC/41276-7', 'SNOMED/25469001'],
        'min_max': [-20, 50],
        'normal_range': [3, 11], # Wikipedia
        'unit': 'mmol/L',
        'format': lambda x: f"{int(x)}"
    }
}
# Codes to aggregated events
CODES_TO_AGGREGATED_EVENTS = {code: event for event, codes in AGGREGATED_EVENTS.items() for code in codes['codes']}
# List of all aggregated codes
AGGREGATED_EVENTS_CODES = [code for special_event in AGGREGATED_EVENTS.values() for code in special_event['codes']]
AGGREGATED_EVENTS_CODES_LOINC = [code for special_event in AGGREGATED_EVENTS.values() for code in special_event['codes'] if 'LOINC' in code]
# For medication entries
MEDICATION_ONTOLOGIES = ['RxNorm', 'RxNorm Extension']


def get_special_events_most_recent(events: List[Event]) -> Dict[str, List[Event]]:
    
    result_events = defaultdict(list)

    for event in events:
        # Some values are contain 'Invalid'
        if event.value is not None and not isinstance(event.value, str):
            aggregated_event = CODES_TO_AGGREGATED_EVENTS.get(event.code, None)
            assert aggregated_event is not None, f"Event code {event.code} not found in aggregated events"
            if event.value >= AGGREGATED_EVENTS[aggregated_event]['min_max'][0] and event.value <= AGGREGATED_EVENTS[aggregated_event]['min_max'][1]:
                result_events[aggregated_event].append(event)
   
    # Sort all events by start time - newest first
    for SPECIAL_EVENT in result_events:
        result_events[SPECIAL_EVENT] = sorted(result_events[SPECIAL_EVENT], key=lambda x: x.start, reverse=True)         
    
    return result_events

def datetime_to_markdown(dt):
    display_date = dt.strftime("%Y-%m-%d %H:%M")
    iso_date = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"[{display_date}]({iso_date})"

def datetime_date_to_markdown(dt):
    display_date = dt.strftime("%Y-%m-%d")
    iso_date = dt.strftime("%Y-%m-%d")
    return f"[{display_date}]({iso_date})"  

def datetimes_to_visit_time(label_dt, visit_dt, duration_days=None):
    days_before_label = (label_dt - visit_dt).days
    if duration_days is not None and duration_days > 0:
        return f"{datetime_date_to_markdown(visit_dt)} ({days_before_label} days before prediction time, duration: {duration_days} days)"
    else:
        return f"{datetime_date_to_markdown(visit_dt)} ({days_before_label} days before prediction time)"

def visit_heading(label_dt: datetime, visit) -> str:
    shifted_visit_dt = CONSTANT_LABEL_TIME - (label_dt - visit.start)
    duration_days = (visit.end - visit.start).days if visit.end is not None else None
    return f"### {visit.description} on {datetimes_to_visit_time(CONSTANT_LABEL_TIME, shifted_visit_dt, duration_days)}\n\n"

class SerializationStrategy(ABC):
    @abstractmethod
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        pass

    def get_time_text(self):
        return f"Current time: {datetime_date_to_markdown(CONSTANT_LABEL_TIME)}\n\n"

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
                unit_str = f" ({event.unit})" if event.unit is not None else ""
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
            # Must be treated different to aggregate numeric values
            return self.serialize_unique_event_list(events, numeric_values)
        else:
            return '\n'.join([self.serialize_event(event, numeric_values) for event in events])

    def serialize_unique_event_list(self, events: List[Event], numeric_values=False, keep_last=None) -> str:
        # This function is more complex than just using the unique events, because it also aggregates numeric values
        event_dict = defaultdict(lambda: {'values': [], 'unit': None})
        for event in events:
            # Events iterated from oldest to newest
            event_dict[event.description]['values'].append(event.value)  # type: ignore
            event_dict[event.description]['unit'] = event.unit
        # Remove all None values
        for description, data in event_dict.items():
            if data['values'] is not None:
                data['values'] = [value for value in data['values'] if value is not None]
                if keep_last is not None:
                    # Select the last keep_last values
                    data['values'] = data['values'][-keep_last:]

        serialized_events = []
        for description, data in event_dict.items():
            if numeric_values:
                values_str = ', '.join(map(self.format_value, data['values'] if data['values'] is not None else []))
                unit_str = f" ({data['unit']})" if data['unit'] is not None else ""
                if values_str:
                    serialized_events.append(f"- {description}{unit_str}: {values_str}")
                else:
                    serialized_events.append(f"- {description}")
            else:
                serialized_events.append(f"- {description}")
            values_str = ', '.join(map(str, data['values'] if data['values'] is not None else []))

        return '\n'.join(serialized_events)

    def list_visits_with_events_by_category(self, ehr_serializer, label_time, numeric_values=False, unique_events=False, keep_last=None) -> str:
        visit_texts = []
        # Set label time to a constant value for all patients
        
        for visit in sorted(ehr_serializer.visits, reverse=True):
            visit_text = visit_heading(label_time, visit)
            
            general_events = [e for e in visit.events if e.code.split('/')[0] not in ['RxNorm', 'RxNorm Extension', 'CPT4', 'ICD10PCS', 'ICD9Proc', 'Visit']]
            medication_events = [e for e in visit.events if e.code.split('/')[0] in ['RxNorm', 'RxNorm Extension']]
            procedure_events = [e for e in visit.events if e.code.split('/')[0] in ['CPT4', 'ICD10PCS', 'ICD9Proc']]
            
            categories = []
            if general_events:
                categories.append("#### Conditions\n\n" + self.serialize_unique_event_list(general_events, numeric_values=numeric_values, keep_last=keep_last))
            if medication_events:
                categories.append("#### Medications\n\n" + self.serialize_unique_event_list(medication_events, numeric_values=numeric_values, keep_last=keep_last))
            if procedure_events:    
                categories.append("#### Procedures\n\n" + self.serialize_unique_event_list(procedure_events, numeric_values=numeric_values, keep_last=keep_last))
            visit_texts.append(visit_text + '\n\n'.join(categories))

        return '\n\n'.join(visit_texts)

    def list_visits_with_events(self, ehr_serializer, label_time, numeric_values=False, unique_events=False) -> str:
        visit_texts = []
        # Set label time to a constant value for all patients
        for visit in sorted(ehr_serializer.visits, reverse=True):
            visit_text = visit_heading(label_time, visit) + self.serialize_event_list(visit.events, numeric_values=numeric_values, unique_events=unique_events)
            visit_texts.append(visit_text)

        return '\n\n'.join(visit_texts)
    
    def _serialize_aggregated_events_list_value(self, type, event, include_date):
        if event.value is None:
            return ''
        
        formatted_value = AGGREGATED_EVENTS[type]['format'](event.value)
        result = f"{formatted_value}"
        
        # Check if has normal range
        if 'normal_range' in AGGREGATED_EVENTS[type]:
                
            rating = "normal"
            if event.value < AGGREGATED_EVENTS[type]['normal_range'][0]:
                rating = "low"
            elif event.value > AGGREGATED_EVENTS[type]['normal_range'][1]:
                rating = "high"
            result += f" ({rating})"
            
        if include_date:
            result += f" - {datetime_to_markdown(event.start)}"
            
        return result

    def serialize_aggregated_events_list(self, aggregated_events, num_values, include_date=False):
        serialization = []
        aggregated_events_recent = get_special_events_most_recent(aggregated_events)
        
        for sub_list in ['Body Metrics', 'Vital Signs', 'Lab Results']:
            serialization.append(AGGREGATED_SUB_EVENTS[sub_list]['heading'])
            event_types = AGGREGATED_SUB_EVENTS[sub_list]['events']
            for event_type in event_types:
                if event_type in aggregated_events_recent:
                    if len(aggregated_events_recent[event_type]) >= num_values:
                        num_values_aggregated_events = aggregated_events_recent[event_type][:num_values]
                    else:
                        num_values_aggregated_events = aggregated_events_recent[event_type]
                    num_values_aggregated_events = [event for event in num_values_aggregated_events if event.value is not None]
                    serialization.append(f"- {event_type} ({AGGREGATED_EVENTS[event_type]['unit']}): " +\
                        ', '.join([self._serialize_aggregated_events_list_value(event_type, event, include_date) for event in num_values_aggregated_events]))
                else:
                    serialization.append(f"- {event_type}: No recent data")
            serialization.append("")
            
        return '\n'.join(serialization) + "\n"
    
    def _serialize_aggregated_events_list_value_detailed(self, type, event, label_time=None):
        if event.value is None:
            return ''
        
        formatted_value = AGGREGATED_EVENTS[type]['format'](event.value)
        
        time = event.start
        if label_time is not None:
            time = CONSTANT_LABEL_TIME - (label_time - event.start)
        result = f"- {datetime_to_markdown(time)}: {type} value was {formatted_value} {AGGREGATED_EVENTS[type]['unit']}"
        
        # Check if has normal range
        if 'normal_range' in AGGREGATED_EVENTS[type]:
                
            rating = "normal"
            if event.value < AGGREGATED_EVENTS[type]['normal_range'][0]:
                rating = "too low"
            elif event.value > AGGREGATED_EVENTS[type]['normal_range'][1]:
                rating = "too high"
                
            result += f", which is {rating}"
                
        return result
    
    def serialize_aggregated_events_list_detailed(self, aggregated_events, num_values, include_date=False, label_time=None):
        serialization = []
        aggregated_events_recent = get_special_events_most_recent(aggregated_events)
        
        for sub_list in ['Body Metrics', 'Vital Signs', 'Lab Results']:
            serialization.append(AGGREGATED_SUB_EVENTS[sub_list]['heading'])
            event_types = AGGREGATED_SUB_EVENTS[sub_list]['events']
            for event_type in event_types:
                if event_type in aggregated_events_recent:
                    if len(aggregated_events_recent[event_type]) >= num_values:
                        num_values_aggregated_events = aggregated_events_recent[event_type][:num_values]
                    else:
                        num_values_aggregated_events = aggregated_events_recent[event_type]
                    num_values_aggregated_events = [event for event in num_values_aggregated_events if event.value is not None]
                    serialization.append(f"### {event_type}\n\n" +\
                        '\n'.join([self._serialize_aggregated_events_list_value_detailed(event_type, event, label_time) for event in num_values_aggregated_events]) + "\n")
                else:
                    serialization.append(f"### {event_type}\n\n" + "No recent data\n")
            
        return '\n'.join(serialization) + "\n"

    def get_medication_last_visit(self, ehr_serializer, numeric_values):
        if ehr_serializer.most_recent_visit_with_medications is None:
            return MEDICATIONS_EVENTS_HEADING + 'No recent medications\n\n'
        
        visit, events = ehr_serializer.most_recent_visit_with_medications
        visit_text = MEDICATIONS_EVENTS_HEADING + self.serialize_unique_event_list(events, numeric_values=numeric_values) + '\n\n'
        return visit_text
    
    def get_unique_values_of_ontologies(self, ehr_serializer, ontologies, negated=False):
        unique_description = set()
        unique_events = []
        
        def check_onotology(event):
                if event.code.split('/')[0] in ontologies:
                    return (not negated)
                return negated
        
        def add_unique_event(event):
            if event.description not in unique_description:
                unique_description.add(event.description)
                unique_events.append(event)
        
        # First check static events
        for event in ehr_serializer.static_events:
            if check_onotology(event):
                add_unique_event(event)
                    
        # Then check visits
        for visit in ehr_serializer.visits:
            for event in visit.events:
                if check_onotology(event):
                    add_unique_event(event)
                        
        return unique_events
    
    def get_demographics_aggr_events_visits_serialization(self, ehr_serializer, label_time: datetime) -> str:
        # Add demographics
        num_demographics = 4
        if len(ehr_serializer.static_events) >= 3 and ehr_serializer.static_events[2].code.startswith('Ethnicity'):
            num_demographics = 3
        if len(ehr_serializer.static_events) <= num_demographics:
            num_demographics = len(ehr_serializer.static_events)
            
        demographics_serialization = "## Patient Demographics\n\n" + self.serialize_unique_event_list(ehr_serializer.static_events[:num_demographics], numeric_values=False) + '\n\n'
        ehr_serializer.static_events = ehr_serializer.static_events[num_demographics:]
        
        # Add aggregated events
        aggr_events_serialization = ""
        if self.num_aggregated_events > 0:
            aggr_events_serialization = self.serialize_aggregated_events_list(ehr_serializer.aggregated_events, self.num_aggregated_events)
        
        # Add visits
        visits_serialization = "## Past Medical Visits\n\n" + '\n'.join(["- " + visit_heading(label_time, visit)[4:-2] for visit in sorted(ehr_serializer.visits, reverse=True)]) + '\n\n'
        
        return demographics_serialization + aggr_events_serialization + visits_serialization
                
class ListEventsStrategy(SerializationStrategy):
    def __init__(self, unique_events: bool, numeric_values: bool, medication_entry: bool, num_aggregated_events: int):
        self.unique_events = unique_events
        self.numeric_values = numeric_values
        self.medication_entry = medication_entry
        self.num_aggregated_events = num_aggregated_events
    
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        aggr_events_serialization = ""
        if self.num_aggregated_events > 0:
            aggr_events_serialization = self.serialize_aggregated_events_list(ehr_serializer.aggregated_events, self.num_aggregated_events)
            
        medication_entry_serialization = ""
        if self.medication_entry:
            medication_entry_serialization = self.get_medication_last_visit(ehr_serializer, numeric_values=self.numeric_values)
            
        events = ehr_serializer.static_events + [event for visit in ehr_serializer.visits for event in visit.events]
        events = sorted(events, key=lambda x: x.start)
        unique_events = self.get_unique_events(events)
        return EHR_HEADING + aggr_events_serialization + medication_entry_serialization + STATIC_EVENTS_HEADING + self.serialize_event_list(unique_events, numeric_values=self.numeric_values, unique_events=self.unique_events)


class UniqueThenListVisitsStrategy(SerializationStrategy):
    def __init__(self, num_aggregated_events: int, medication_entry: bool):
        self.num_aggregated_events = num_aggregated_events
        self.medication_entry = medication_entry
    
    def serialize(self, ehr_serializer, label_time: datetime) -> str:

        demographics_aggr_events_visits_serialization = self.get_demographics_aggr_events_visits_serialization(ehr_serializer, label_time)
        
        # Add general events
        general_events_serialization = "## All Medical Conditions\n\n" + self.serialize_unique_event_list(
            self.get_unique_values_of_ontologies(ehr_serializer, ['RxNorm', 'RxNorm Extension', 'CPT4', 'ICD10PCS', 'ICD9Proc', 'Visit'], negated=True),
            numeric_values=False
        ) + '\n\n'
        
        # Detailed Medical History
        medical_history = "## Detailed Past Medical Visits (most recent first)\n\n" + self.list_visits_with_events_by_category(ehr_serializer, label_time, numeric_values=False, unique_events=True) 
        
        result = EHR_HEADING + self.get_time_text() + demographics_aggr_events_visits_serialization + general_events_serialization + medical_history
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result

class UniqueThenListVisitsWithValuesStrategy(SerializationStrategy):
    def __init__(self, num_aggregated_events: int, medication_entry: bool):
        self.num_aggregated_events = num_aggregated_events
        self.medication_entry = medication_entry
    
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        
        demographics_aggr_events_visits_serialization = self.get_demographics_aggr_events_visits_serialization(ehr_serializer, label_time)

        # Add general events
        general_events_serialization = "## All Medical Conditions\n\n" + self.serialize_unique_event_list(
            self.get_unique_values_of_ontologies(ehr_serializer, ['RxNorm', 'RxNorm Extension', 'CPT4', 'ICD10PCS', 'ICD9Proc', 'Visit'], negated=True),
            numeric_values=True,
            keep_last=3
        ) + '\n\n'
        
        # Detailed Medical History
        medical_history = "## Detailed Past Medical Visits (most recent first)\n\n" +\
            self.list_visits_with_events_by_category(ehr_serializer, label_time, numeric_values=True, unique_events=True, keep_last=3) 
        
        result = EHR_HEADING + self.get_time_text() + demographics_aggr_events_visits_serialization + general_events_serialization + medical_history
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result

class UniqueThenListVisitsWOAllCondsStrategy(SerializationStrategy):
    def __init__(self, num_aggregated_events: int, medication_entry: bool):
        self.num_aggregated_events = num_aggregated_events
        self.medication_entry = medication_entry
    
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        
        demographics_aggr_events_visits_serialization = self.get_demographics_aggr_events_visits_serialization(ehr_serializer, label_time)

        # Add general events
        general_events_serialization = STATIC_EVENTS_HEADING + self.serialize_unique_event_list(ehr_serializer.static_events, numeric_values=False) + '\n\n'
        
        # Detailed Medical History
        medical_history = "## Detailed Past Medical Visits (most recent first)\n\n" + self.list_visits_with_events_by_category(ehr_serializer, label_time, numeric_values=False, unique_events=True) 
        
        result = EHR_HEADING + self.get_time_text() + demographics_aggr_events_visits_serialization + general_events_serialization + medical_history
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result

class UniqueThenListVisitsWOAllCondsWithValuesStrategy(SerializationStrategy):
    def __init__(self, num_aggregated_events: int, medication_entry: bool):
        self.num_aggregated_events = num_aggregated_events
        self.medication_entry = medication_entry
    
    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        
        demographics_aggr_events_visits_serialization = self.get_demographics_aggr_events_visits_serialization(ehr_serializer, label_time)

        # Add general events
        general_events_serialization = STATIC_EVENTS_HEADING + self.serialize_unique_event_list(ehr_serializer.static_events, numeric_values=True, keep_last=3) + '\n\n'
        
        # Detailed Medical History
        medical_history = "## Detailed Past Medical Visits (most recent first)\n\n" +\
            self.list_visits_with_events_by_category(ehr_serializer, label_time, numeric_values=True, unique_events=True, keep_last=3) 
        
        result = EHR_HEADING + self.get_time_text() + demographics_aggr_events_visits_serialization + general_events_serialization + medical_history
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result

class ListVisitsWithEventsStrategy(SerializationStrategy):
    def __init__(self, unique_events: bool, numeric_values: bool, medication_entry: bool, num_aggregated_events: int):
        self.unique_events = unique_events
        self.numeric_values = numeric_values
        self.medication_entry = medication_entry
        self.num_aggregated_events = num_aggregated_events

    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        aggr_events_serialization = ""
        if self.num_aggregated_events > 0:
            aggr_events_serialization = self.serialize_aggregated_events_list(ehr_serializer.aggregated_events, self.num_aggregated_events)
       
        medication_entry_serialization = ""
        if self.medication_entry:
            medication_entry_serialization = self.get_medication_last_visit(ehr_serializer, numeric_values=self.numeric_values)
            
        if self.unique_events:
            static_text = STATIC_EVENTS_HEADING + self.serialize_unique_event_list(ehr_serializer.static_events, numeric_values=self.numeric_values)
        else:
            static_text = STATIC_EVENTS_HEADING + self.serialize_event_list(ehr_serializer.static_events, numeric_values=self.numeric_values, unique_events=False)
        visits_text = VISITS_EVENTS_HEADING + self.list_visits_with_events(ehr_serializer, label_time, numeric_values=self.numeric_values, unique_events=self.unique_events) 
        return EHR_HEADING + self.get_time_text() + static_text + '\n\n' + aggr_events_serialization + medication_entry_serialization + visits_text

class ListVisitsWithEventsDetailedAggrStrategy(SerializationStrategy):
    def __init__(self, unique_events: bool, numeric_values: bool, medication_entry: bool, num_aggregated_events: int):
        self.unique_events = unique_events
        self.numeric_values = numeric_values
        self.medication_entry = medication_entry
        self.num_aggregated_events = num_aggregated_events

    def serialize(self, ehr_serializer, label_time: datetime) -> str:
        aggr_events_serialization = ""
        if self.num_aggregated_events > 0:
            aggr_events_serialization = self.serialize_aggregated_events_list_detailed(ehr_serializer.aggregated_events, self.num_aggregated_events, include_date=False, label_time=label_time)
       
        medication_entry_serialization = ""
        if self.medication_entry:
            medication_entry_serialization = self.get_medication_last_visit(ehr_serializer, numeric_values=self.numeric_values)
            
        if self.unique_events:
            static_text = STATIC_EVENTS_HEADING + self.serialize_unique_event_list(ehr_serializer.static_events, numeric_values=self.numeric_values)
        else:
            static_text = STATIC_EVENTS_HEADING + self.serialize_event_list(ehr_serializer.static_events, numeric_values=self.numeric_values, unique_events=False)
        visits_text = VISITS_EVENTS_HEADING + self.list_visits_with_events(ehr_serializer, label_time, numeric_values=self.numeric_values, unique_events=self.unique_events) 
        return EHR_HEADING + self.get_time_text() + static_text + '\n\n' + aggr_events_serialization + medication_entry_serialization + visits_text

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
        code: Optional[str] = None
     ):
        self.start = start
        self.end = end
        self.description = description
        self.value = value
        self.unit = unit
        self.code = code

class EHRSerializer:
    def __init__(self):
        self.visits: List[EHRVisit] = []
        self.static_events: List[EHREvent] = []
        self.aggregated_events: List[Event] = []
        self.most_recent_visit_with_medications: Optional[Tuple[EHRVisit, List[EHREvent]]] = None
                
    def _parse_visit(self, event_visit: Event, description: str) -> EHRVisit:
        return EHRVisit(
            visit_id=event_visit.visit_id,
            start=event_visit.start,
            end=event_visit.end if hasattr(event_visit, 'end') else None,
            description=description
        )
        
    def _parse_event(self, event: Event, description: str) -> EHREvent:
        return EHREvent(
            start=event.start,
            end=event.end if hasattr(event, 'end') else None,
            description=description,
            value=event.value if hasattr(event, 'value') else None,
            unit=event.unit if hasattr(event, 'unit') else None,
            code=event.code if hasattr(event, 'code') else None
        )
                
    def load_from_femr_events(self, events: List[Event], resolve_code, is_visit_event: Callable[[Event], bool], filter_aggregated_events) -> None:
        
        # Filter aggregated events that are treated separately
        # Do so when num_aggregated_events > 0, i.e. they should be displayed
        if filter_aggregated_events:
            non_aggregated_events = []
            for event in events:
                if event.code not in AGGREGATED_EVENTS_CODES:
                    non_aggregated_events.append(event)
                else:
                    self.aggregated_events.append(event)
            events = non_aggregated_events
              
        # For medication entry keep track of visits with medications and visit starts
        visit_start_times = []
        visits_with_medications = set()

        # First process all visits
        visit_ids_to_visits: Dict[int, EHRVisit] = {}
        for event_visit in filter(is_visit_event, events):
            description = resolve_code(event_visit.code)
            if description is not None:
                visit = self._parse_visit(event_visit, description)
                visit_ids_to_visits[event_visit.visit_id] = visit
                visit_start_times.append([event_visit.start, visit])
        
        # Then process all events
        for event in filter(lambda x: not is_visit_event(x), events):
            visit = visit_ids_to_visits.get(event.visit_id, None)
            if event.code.startswith(tuple(MEDICATION_ONTOLOGIES)) and event.visit_id is not None:
                visits_with_medications.add(event.visit_id)
            description = resolve_code(event.code)
            if description is not None:
                event = self._parse_event(event, description)
                if visit is not None:
                    visit.add_event(event)
                else:
                    self.static_events.append(event)
                    
        self.visits = sorted(visit_ids_to_visits.values())
        
        # Find earlierst visit with medication and store them
        recent_medication_visit = None
        visit_start_times.sort(key=lambda x: x[0], reverse=True)
        for _, visit in visit_start_times:
            if visit.visit_id in visits_with_medications:
                recent_medication_visit = visit
                break
        
        if recent_medication_visit is None:
            self.most_recent_visit_with_medications = None
        else:
            visit_medication_events = [e for e in events if (not is_visit_event(e)) and (e.visit_id == recent_medication_visit.visit_id) and (e.code.startswith(tuple(MEDICATION_ONTOLOGIES)))]
            visit_medication_ehr_events = []
            for event in visit_medication_events:
                description = resolve_code(event.code, included_ontologies=MEDICATION_ONTOLOGIES)
                if description is not None:
                    visit_medication_ehr_events.append(self._parse_event(event, description))
            self.most_recent_visit_with_medications = (recent_medication_visit, visit_medication_ehr_events)
            
    def set_serialization_strategy(self, serialization_strategy: SerializationStrategy):
        self._serialization_strategy = serialization_strategy

    def serialize(self, serialization_strategy: SerializationStrategy, label_time: datetime) -> str:
        return serialization_strategy.serialize(self, label_time)