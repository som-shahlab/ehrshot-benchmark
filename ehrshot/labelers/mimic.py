import datetime
from typing import List, Tuple
from femr import Patient, Event
from femr.extension import datasets as extension_datasets
from ehrshot.labelers.core import Label, Labeler, LabelType
from ehrshot.labelers.omop import (
    WithinVisitLabeler,
    get_inpatient_admission_events,
    move_datetime_to_end_of_day,
    get_inpatient_admission_discharge_times,
    get_femr_codes,
    identity,
)

class Mimic_ReadmissionLabeler(Labeler):
    """30-day readmissions prediction task.
    Binary prediction task @ 11:59PM on the day of discharge whether the patient will be readmitted within 30 days.
    
    Excludes:
        - Readmissions that occurred on the same day
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.time_horizon: datetime.timedelta = datetime.timedelta(days=30)
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def label(self, patient: Patient) -> List[Label]:
        labels: List[Label] = []
        times: List[Tuple[datetime.datetime]] = get_inpatient_admission_discharge_times(patient, self.ontology)
        admission_times = sorted([ x[0] for x in times ])
        for idx, admission_time in enumerate(admission_times):
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(admission_time)
            is_30_day_readmission = False
            for admission_time2 in admission_times[idx + 1:]:
                # Ignore readmissions that occur on before or on same day as prediction time
                if admission_time2 <= prediction_time:
                    continue
                # If readmission occurs within 30 days, mark as True
                if (admission_time2 - prediction_time) <= self.time_horizon:
                    is_30_day_readmission: bool = True
                    break
            labels.append(Label(prediction_time, is_30_day_readmission))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class Mimic_MortalityLabeler(WithinVisitLabeler):
    """In-hospital mortality prediction task.

    Binary prediction task @ 11:59PM on the day of admission whether the patient dies during their hospital stay.

    Excludes:
        - Admissions with no length-of-stay (i.e. `event.end is None` )
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day
        self.visit_start_adjust_func = identity
        self.visit_end_adjust_func = identity

    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Return a list of all visits.

        Excludes:
            - Admissions with no length-of-stay (i.e. `event.end is None` )
        """
        visits, visit_idxs = get_inpatient_admission_events(patient, self.ontology, is_return_idx=True)  # type: ignore
        valid_events: List[Event] = []
        for e in visits:
            if (
                e.end is not None
            ):
                valid_events.append(e)
        return valid_events

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome (i.e., death)."""
        death_concepts: List[str] = [ "SNOMED/419620001", ]
        outcome_codes: List[str] = list(get_femr_codes(self.ontology, death_concepts, is_ontology_expansion=True))
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in outcome_codes:
                times.append(e.start)
        return times

    def get_labeler_type(self) -> LabelType:
        return "boolean"

if __name__ == '__main__':
    pass