# # Import necessary libraries and modules
# import os
# from tqdm import tqdm
# import pandas as pd
# import numpy as np
# import datetime
# from collections import deque, defaultdict
# import time
# import random
# from datetime import timedelta
# import csv
# import femr.datasets
# from typing import List, Tuple, Set
# from femr import Patient
# from femr.extension import datasets as extension_datasets
# from hf_ehr.data.datasets import FEMRDataset
# from ehrshot.labelers.core import Label, Labeler, LabelType, TimeHorizon, TimeHorizonEventLabeler
# from ehrshot.labelers.omop import (
#     CodeLabeler,
#     WithinVisitLabeler,
#     get_death_concepts,
#     get_inpatient_admission_events,
#     move_datetime_to_end_of_day,
# )

# BIRTH_CODE: str = "SNOMED/3950001"

# def calculate_age(birthdate: datetime.datetime, event_time: datetime.datetime) -> int:
#     """Return age (in years)"""
#     return event_time.year - birthdate.year

# def _get_all_children(ontology: extension_datasets.Ontology, code: str) -> Set[str]:
#     children_code_set = set([code])
#     parent_deque = deque([code])

#     while len(parent_deque) > 0:
#         temp_parent_code: str = parent_deque.popleft()
#         try:
#             for temp_child_code in ontology.get_children(temp_parent_code):
#                 children_code_set.add(temp_child_code)
#                 parent_deque.append(temp_child_code)
#         except:
#             pass

#     return children_code_set

# def get_femr_codes(
#     ontology: extension_datasets.Ontology,
#     omop_concept_codes: List[str],
#     is_ontology_expansion: bool = True,
#     is_silent_not_found_error: bool = True,
# ) -> Set[str]:
#     """Does ontology expansion on the given OMOP concept codes if `is_ontology_expansion` is True."""
#     if not isinstance(omop_concept_codes, list):
#         omop_concept_codes = [omop_concept_codes]
#     codes: Set[str] = set()
#     for omop_concept_code in omop_concept_codes:
#         try:
#             expanded_codes = (
#                 _get_all_children(ontology, omop_concept_code) if is_ontology_expansion else {omop_concept_code}
#             )
#             codes.update(expanded_codes)
#         except ValueError:
#             if not is_silent_not_found_error:
#                 raise ValueError(f"OMOP Concept Code {omop_concept_code} not found in ontology.")
#     return codes

# def get_inpatient_admission_concepts() -> List[str]:
#     return ["Visit/IP", "Visit/ERIP"]

# def get_inpatient_admission_codes(ontology: extension_datasets.Ontology) -> Set[int]:
#     # Don't get children here b/c it adds noise (i.e. "Medicare Specialty/AO")
#     return get_femr_codes(
#         ontology, get_inpatient_admission_concepts(), is_ontology_expansion=False, is_silent_not_found_error=True
#     )

# def get_inpatient_admission_events(patient: Patient, ontology: extension_datasets.Ontology) -> List[Event]:
#     admission_codes: Set[str] = get_inpatient_admission_codes(ontology)
#     events: List[Event] = []
#     for e in patient.events:
#         if e.code in admission_codes and e.omop_table == "visit_occurrence":
#             # Error checking
#             if e.start is None or e.end is None:
#                 raise RuntimeError(f"Event {e} cannot have `None` as its `start` or `end` attribute.")
#             elif e.start > e.end:
#                 raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
#             # Drop single point in time events
#             if e.start == e.end:
#                 continue
#             events.append(e)
#     return events


# def get_inpatient_admission_discharge_times(
#     patient: Patient, ontology: extension_datasets.Ontology
# ) -> List[Tuple[datetime.datetime, datetime.datetime]]:
#     """Return a list of all admission/discharge times for this patient."""
#     events: List[Event] = get_inpatient_admission_events(patient, ontology)
#     times: List[Tuple[datetime.datetime, datetime.datetime]] = []
#     for e in events:
#         if e.end is None:
#             raise RuntimeError(f"Event {e} cannot have `None` as its `end` attribute.")
#         if e.start > e.end:
#             raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
#         times.append((e.start, e.end))
#     return times

# # class Mimic4LongLOSLabeler(Labeler):
# #     """Long LOS prediction task

# #     Binary prediction task @ 11:59PM on the day of admission whether the patient stays in hospital for >=7 days
    
# #     Excludes:
# #         - Visits where discharge occurs on the same day as admission
# #         - Visits where patient <= 18 years of age on admission
# #     """

# #     def __init__(
# #         self,
# #         ontology: extension_datasets.Ontology,
# #     ):
# #         self.ontology: extension_datasets.Ontology = ontology
# #         self.long_time: datetime.timedelta = datetime.timedelta(days=7)
# #         self.prediction_time_adjustment_func = move_datetime_to_end_of_day
    
# #     def label(self, patient: Patient) -> List[Label]:
# #         """Label the selected admission with admission length >= `self.long_time`."""
# #         labels: List[Label] = []
# #         birthdate: datetime.datetime = patient.events[0].time
# #         assert patient.events[0].code == BIRTH_CODE, f"Patient {patient} doesn't have a birth code {BIRTH_CODE} as their first event."
        
# #         for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
# #             # If admission and discharge are on the same day, then ignore
# #             if admission_time.date() == discharge_time.date():
# #                 continue
# #             # If patient <= 18 yrs of age at admission, then ignore
# #             if calculate_age(birthdate, admission_time) <= 18:
# #                 continue
# #             is_long_admission: bool = (discharge_time - admission_time) >= self.long_time
# #             prediction_time: datetime.datetime = self.prediction_time_adjustment_func(admission_time)
# #             labels.append(Label(prediction_time, is_long_admission))
        
# #         return labels

# #     def get_labeler_type(self) -> LabelType:
# #         return "boolean"

# # class Mimic4Readmission30DayLabeler(TimeHorizonEventLabeler):
# #     """30-day readmissions prediction task.

# #     Binary prediction task @ 11:59PM on the day of discharge whether the patient will be readmitted within 30 days.
        
# #     Excludes:
# #         - Patients readmitted on same day as discharge
# #         - Visits where patient <= 18 years of age on admission
# #     """

# #     def __init__(
# #         self,
# #         ontology: extension_datasets.Ontology,
# #     ):
# #         self.ontology = ontology
# #         self.time_horizon = TimeHorizon(
# #             start=datetime.timedelta(minutes=1), end=datetime.timedelta(days=30)
# #         )
# #         self.prediction_time_adjustment_func = move_datetime_to_end_of_day

# #     def get_outcome_times(self, patient: Patient, selected_discharge_time: datetime.datetime) -> List[datetime.datetime]:
# #         """Return the start times of inpatient admissions that occur after the selected discharge time."""
# #         times: List[datetime.datetime] = []
# #         for admission_time, __ in get_inpatient_admission_discharge_times(patient, self.ontology):
# #             if admission_time > selected_discharge_time:  # Ensure only subsequent admissions are considered
# #                 times.append(admission_time)
# #         return times

# #     def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
# #         """Return end of admission as prediction timm."""
# #         times: List[datetime.datetime] = []
# #         admission_times = set()
# #         birthdate: datetime.datetime = patient.events[0].time
# #         assert patient.events[0].code == BIRTH_CODE, f"Patient {patient} doesn't have a birth code {BIRTH_CODE} as their first event."
        
# #         for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
# #             prediction_time: datetime.datetime = self.prediction_time_adjustment_func(discharge_time)
# #             # Ignore patients who are readmitted the same day they were discharged b/c of data leakage
# #             if prediction_time.replace(hour=0, minute=0, second=0, microsecond=0) in admission_times:
# #                 continue
# #             # If patient <= 18 yrs of age at admission, then ignore
# #             if calculate_age(birthdate, admission_time) <= 18:
# #                 continue
# #             times.append(prediction_time)
# #             admission_times.add(admission_time.replace(hour=0, minute=0, second=0, microsecond=0))
# #         times = sorted(list(set(times)))
# #         return times

# #     def get_time_horizon(self) -> TimeHorizon:
# #         return self.time_horizon

# #     def get_labeler_type(self) -> LabelType:
# #         return "boolean"


# class Mimic_MortalityLabeler(WithinVisitLabeler):
#     """In-hospital mortality prediction task.

#     Binary prediction task @ 11:59PM on the day of admission whether the patient dies during their hospital stay.

#     Excludes:
#         - Admissions with no length-of-stay (i.e. `event.end is None` )
#         - Admissions < 48 hours
#         - Admissions with no events before 48 hours
#     """

#     def __init__(
#         self,
#         ontology: extension_datasets.Ontology,
#     ):
#         self.ontology: extension_datasets.Ontology = ontology
#         self.prediction_time_adjustment_func = move_datetime_to_end_of_day

#     def get_visit_events(self, patient: Patient) -> List[Event]:
#         """Return a list of all visits.

#         Excludes:
#             - Admissions with no length-of-stay (i.e. `event.end is None` )
#             - Admissions < 48 hours
#             - Admissions with no events before 48 hours
#         """
#         visits: List[Event] = get_inpatient_admission_events(patient, self.ontology, is_return_idx=False)  # type: ignore
#         valid_events: List[Event] = []
#         for __, e in visits:
#             if (
#                 e.end is not None
#                 and e.end - e.start >= datetime.timedelta(hours=48)
#                 and does_exist_event_within_time_range(
#                     patient, e.start, e.start + datetime.timedelta(hours=48), exclude_event_idxs=icu_event_idxs
#                 )
#             ):
#                 valid_events.append(e)
#         return valid_events

#     def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
#         """Return a list of all times when the patient experiences an outcome (i.e., death)."""
#         outcome_codes = list(get_femr_codes(self.ontology, get_death_concepts(), is_ontology_expansion=True))
#         times: List[datetime.datetime] = []
#         for e in patient.events:
#             if e.code in outcome_codes:
#                 times.append(e.start)
#         return times

#     def get_labeler_type(self) -> LabelType:
#         return "boolean"



# if __name__ == '__main__':
#     from hf_ehr.config import PATH_TO_FEMR_EXTRACT_MIMIC4  # Ensure correct path

#     # Load the FEMR dataset
#     splits = ['train', 'val', 'test']
    
#     # Load the ontology
#     ontology = FEMRDataset(PATH_TO_FEMR_EXTRACT_MIMIC4, split='train').femr_db.get_ontology()

#     # Initialize the labelers
#     labelers = {
#         "LongLOS": Mimic4LongLOSLabeler(ontology=ontology),
#         "Readmission30Day": Mimic4Readmission30DayLabeler(ontology=ontology),
#         "Mortality": Mimic4MortalityLabeler(ontology=ontology),
#     }

#     # Initialize counters for total results across all splits
#     total_labeler_stats = {labeler_name: {"total_labels": 0, "positive_labels": 0} for labeler_name in labelers.keys()}

#     # Process each split
#     for split_name in splits:
#         dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT_MIMIC4, split=split_name)
        
#         # Get the patient IDs
#         patient_ids = dataset.get_pids()
        
#         # Add tqdm progress bar for processing patients
#         for pid in tqdm(patient_ids, desc=f"Processing split: {split_name}"):
#             patient = dataset.femr_db[pid]

#             # Get all inpatient admissions for the patient
#             admission_times = get_inpatient_admission_discharge_times(patient, ontology)
            
#             # Filter out admissions where the patient was less than 18 years old or where admission and discharge were on the same day
#             snomed_event_time = next((event.start for event in patient.events if event.code == "SNOMED/3950001"), None)
#             birthdate = snomed_event_time

#             filtered_admissions = [
#                 (admission_time, discharge_time)
#                 for admission_time, discharge_time in admission_times
#                 if birthdate is not None and calculate_age(birthdate, admission_time) >= 18 and admission_time.date() != discharge_time.date()
#             ]
            
#             if not filtered_admissions:
#                 continue
            
#             # Randomly select one admission from the filtered list
#             selected_admission_time, selected_discharge_time = random.choice(filtered_admissions)

#             # Apply each labeler to the selected admission
#             for labeler_name, labeler in labelers.items():
#                 labels = labeler.label(patient, selected_admission_time, selected_discharge_time)
                
#                 # Count total and positive labels
#                 total_labeler_stats[labeler_name]["total_labels"] += len(labels)
#                 total_labeler_stats[labeler_name]["positive_labels"] += sum(label.value for label in labels)

#     # Print final results in a table format
#     print(f"| {'Task':<20} | {'# of Total Labels':<20} | {'# of Positive Labels':<20} |")
#     print(f"|{'-'*22}|{'-'*22}|{'-'*22}|")
#     for labeler_name, stats in total_labeler_stats.items():
#         print(f"| {labeler_name:<20} | {stats['total_labels']:<20} | {stats['positive_labels']:<20} |")