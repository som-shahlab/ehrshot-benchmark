"""Labeling functions for OMOP data."""
import datetime
from abc import abstractmethod
from collections import deque
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from femr import Patient, Event
from femr.extension import datasets as extension_datasets
from ehrshot.labelers.core import Label, Labeler, LabelType, TimeHorizon, TimeHorizonEventLabeler

def identity(x: Any) -> Any:
    return x


def get_visit_concepts() -> List[str]:
    return ["Visit/IP", "Visit/OP"]


def get_inpatient_admission_concepts() -> List[str]:
    return ["Visit/IP",]


def get_outpatient_visit_concepts() -> List[str]:
    return ["Visit/OP"]

def get_death_concepts() -> List[str]:
    return [
        "Condition Type/OMOP4822053",
    ]

def get_icu_visit_detail_concepts() -> List[str]:
    return [
        # All care sites with "ICU" (case insensitive) in the name
        "CARE_SITE/7928450",
        "CARE_SITE/7930385",
        "CARE_SITE/7930600",
        "CARE_SITE/7928852",
        "CARE_SITE/7928619",
        "CARE_SITE/7929727",
        "CARE_SITE/7928675",
        "CARE_SITE/7930225",
        "CARE_SITE/7928759",
        "CARE_SITE/7928227",
        "CARE_SITE/7928810",
        "CARE_SITE/7929179",
        "CARE_SITE/7928650",
        "CARE_SITE/7929351",
        "CARE_SITE/7928457",
        "CARE_SITE/7928195",
        "CARE_SITE/7930681",
        "CARE_SITE/7930670",
        "CARE_SITE/7930176",
        "CARE_SITE/7931420",
        "CARE_SITE/7929149",
        "CARE_SITE/7930857",
        "CARE_SITE/7931186",
        "CARE_SITE/7930934",
        "CARE_SITE/7930924",
    ]


def move_datetime_to_end_of_day(date: datetime.datetime) -> datetime.datetime:
    return date.replace(hour=23, minute=59, second=59)


def does_exist_event_within_time_range(
    patient: Patient, start: datetime.datetime, end: datetime.datetime, exclude_event_idxs: List[int] = []
) -> bool:
    """Return True if there is at least one event within the given time range for this patient.
    If `exclude_event_idxs` is provided, exclude events with those indexes in `patient.events` from the search."""
    excluded = set(exclude_event_idxs)
    for idx, e in enumerate(patient.events):
        if idx in excluded:
            continue
        if start <= e.start <= end:
            return True
    return False


def get_femr_codes(
    ontology: extension_datasets.Ontology,
    omop_concept_codes: List[str],
    is_ontology_expansion: bool = True,
    is_silent_not_found_error: bool = True,
) -> Set[str]:
    """Does ontology expansion on the given OMOP concept codes if `is_ontology_expansion` is True,
        otherwise just returns the codes as given.

    If `is_silent_not_found_error` is True, then this function will NOT raise an error if a given OMOP concept ID is not found in the ontology.
    """
    if not isinstance(omop_concept_codes, list):
        # Make sure a list is passed in
        omop_concept_codes = [omop_concept_codes]
    codes: Set[str] = set()
    for omop_concept_code in omop_concept_codes:
        try:
            codes.update(
                _get_all_children(ontology, omop_concept_code) if is_ontology_expansion else {omop_concept_code}
            )
        except ValueError:
            if not is_silent_not_found_error:
                raise ValueError(f"OMOP Concept Code {omop_concept_code} not found in ontology.")
    return codes


def get_visit_codes(ontology: extension_datasets.Ontology) -> Set[int]:
    return get_femr_codes(ontology, get_visit_concepts(), is_ontology_expansion=True, is_silent_not_found_error=True)


def get_icu_visit_detail_codes(ontology: extension_datasets.Ontology) -> Set[int]:
    return get_femr_codes(
        ontology, get_icu_visit_detail_concepts(), is_ontology_expansion=True, is_silent_not_found_error=True
    )


def get_inpatient_admission_codes(ontology: extension_datasets.Ontology) -> Set[int]:
    # Don't get children here b/c it adds noise (i.e. "Medicare Specialty/AO")
    return get_femr_codes(
        ontology, get_inpatient_admission_concepts(), is_ontology_expansion=False, is_silent_not_found_error=True
    )


def get_outpatient_visit_codes(ontology: extension_datasets.Ontology) -> Set[int]:
    return get_femr_codes(
        ontology, get_outpatient_visit_concepts(), is_ontology_expansion=False, is_silent_not_found_error=True
    )


def get_icu_events(
    patient: Patient, ontology: extension_datasets.Ontology, is_return_idx: bool = False
) -> Union[List[Event], List[Tuple[int, Event]]]:
    """Return all ICU events for this patient.
    If `is_return_idx` is True, then return a list of tuples (event, idx) where `idx` is the index of the event in `patient.events`.
    """
    icu_visit_detail_codes: Set[int] = get_icu_visit_detail_codes(ontology)
    events: Union[List[Event], List[Tuple[int, Event]]] = []
    for idx, e in enumerate(patient.events):
        # `visit_detail` is more accurate + comprehensive than `visit_occurrence` for ICU events for STARR OMOP for some reason
        if e.code in icu_visit_detail_codes and e.omop_table == "visit_detail":
            # Error checking
            if e.start is None or e.end is None:
                raise RuntimeError(
                    f"Event {e} for patient {patient.patient_id} cannot have `None` as its `start` or `end` attribute."
                )
            elif e.start > e.end:
                raise RuntimeError(f"Event {e} for patient {patient.patient_id} cannot have `start` after `end`.")
            # Drop single point in time events
            if e.start == e.end:
                continue
            if is_return_idx:
                events.append((idx, e))  # type: ignore
            else:
                events.append(e)
    return events


def get_outpatient_visit_events(patient: Patient, ontology: extension_datasets.Ontology) -> List[Event]:
    admission_codes: Set[int] = get_outpatient_visit_codes(ontology)
    events: List[Event] = []
    for e in patient.events:
        if e.code in admission_codes and e.omop_table == "visit_occurrence":
            # Error checking
            if e.start is None or e.end is None:
                raise RuntimeError(f"Event {e} cannot have `None` as its `start` or `end` attribute.")
            elif e.start > e.end:
                raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
            # Drop single point in time events
            if e.start == e.end:
                continue
            events.append(e)
    return events


def get_inpatient_admission_events(patient: Patient, ontology: extension_datasets.Ontology, is_return_idx: bool = False) -> List[Event]:
    admission_codes: Set[str] = get_inpatient_admission_codes(ontology)
    events: List[Event] = []
    idxs: List[int] = []
    for idx, e in enumerate(patient.events):
        if e.code in admission_codes and e.omop_table == "visit_occurrence":
            # Error checking
            if e.start is None or e.end is None:
                raise RuntimeError(f"Event {e} cannot have `None` as its `start` or `end` attribute.")
            elif e.start > e.end:
                raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
            # Drop single point in time events
            if e.start == e.end:
                continue
            events.append(e)
            idxs.append(idx)
    if is_return_idx:
        return events, idxs
    return events


def get_inpatient_admission_discharge_times(
    patient: Patient, ontology: extension_datasets.Ontology
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Return a list of all admission/discharge times for this patient."""
    events: List[Event] = get_inpatient_admission_events(patient, ontology)
    times: List[Tuple[datetime.datetime, datetime.datetime]] = []
    for e in events:
        if e.end is None:
            raise RuntimeError(f"Event {e} cannot have `None` as its `end` attribute.")
        if e.start > e.end:
            raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
        times.append((e.start, e.end))
    return times


def _get_all_children(ontology: extension_datasets.Ontology, code: str) -> Set[str]:
    children_code_set = set([code])
    parent_deque = deque([code])

    while len(parent_deque) > 0:
        temp_parent_code: str = parent_deque.popleft()
        try:
            for temp_child_code in ontology.get_children(temp_parent_code):
                children_code_set.add(temp_child_code)
                parent_deque.append(temp_child_code)
        except:
            # The `temp_parent_code` was not found in the ontology, so skip
            pass
    return children_code_set


##########################################################
##########################################################
# Abstract classes derived from Labeler
##########################################################
##########################################################


class WithinVisitLabeler(Labeler):
    """
    The `WithinVisitLabeler` predicts whether or not a patient experiences a specific event
    (as returned by `self.get_outcome_times()`) within each visit.

    Very similar to `TimeHorizonLabeler`, except here we use visits themselves as our time horizon.

    Prediction Time: Start of each visit (adjusted by `self.prediction_adjustment_timedelta` if provided)
    Time horizon: By end of visit
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        visit_start_adjust_func: Optional[Callable] = None,
        visit_end_adjust_func: Optional[Callable] = None,
    ):
        """The argument `visit_start_adjust_func` is a function that takes in a `datetime.datetime`
        and returns a different `datetime.datetime`."""
        self.ontology: extension_datasets.Ontology = ontology
        self.visit_start_adjust_func: Callable = visit_start_adjust_func if visit_start_adjust_func else identity
        self.visit_end_adjust_func: Callable = visit_end_adjust_func if visit_end_adjust_func else identity

    @abstractmethod
    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome"""
        return []

    @abstractmethod
    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Return a list of all visits we want to consider (useful for limiting to inpatient visits)."""
        return []

    def label(self, patient: Patient) -> List[Label]:
        """
        Label all visits returned by `self.get_visit_events()` with whether the patient
        experiences an outcome in `self.outcome_codes` during each visit.
        """
        visits: List[Event] = self.get_visit_events(patient)
        prediction_start_times: List[datetime.datetime] = [
            self.visit_start_adjust_func(visit.start) for visit in visits
        ]
        prediction_end_times: List[datetime.datetime] = [self.visit_end_adjust_func(visit.end) for visit in visits]
        outcome_times: List[datetime.datetime] = self.get_outcome_times(patient)

        # For each visit, check if there is an outcome which occurs within the (start, end) of the visit
        results: List[Label] = []
        curr_outcome_idx: int = 0
        for prediction_idx, (prediction_start, prediction_end) in enumerate(
            zip(prediction_start_times, prediction_end_times)
        ):
            # Error checking
            if curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] is None:
                raise RuntimeError(
                    "Outcome times must be of type `datetime.datetime`, but value of `None`"
                    " provided for `self.get_outcome_times(patient)[{curr_outcome_idx}]"
                )
            if prediction_start is None:
                raise RuntimeError(
                    "Prediction start times must be of type `datetime.datetime`, but value of `None`"
                    " provided for `prediction_start_time`"
                )
            if prediction_end is None:
                raise RuntimeError(
                    "Prediction end times must be of type `datetime.datetime`, but value of `None`"
                    " provided for `prediction_end_time`"
                )
            if prediction_start > prediction_end:
                raise RuntimeError(
                    "Prediction start time must be before prediction end time, but `prediction_start_time`"
                    f" is `{prediction_start}` and `prediction_end_time` is `{prediction_end}`."
                    " Maybe you `visit_start_adjust_func()` or `visit_end_adjust_func()` in such a way that"
                    " the `start` time got pushed after the `end` time?"
                    " For reference, the original state time of this visit is"
                    f" `{visits[prediction_idx].start}` and the original end time is `{visits[prediction_idx].end}`."
                    f" This is for patient with patient_id `{patient.patient_id}`."
                )
            # Find the first outcome that occurs after this visit starts
            # (this works b/c we assume visits are sorted by `start`)
            while curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] < prediction_start:
                # `curr_outcome_idx` is the idx in `outcome_times` that corresponds to the first
                # outcome EQUAL or AFTER the visit for this prediction time starts (if one exists)
                curr_outcome_idx += 1

            # TRUE if an event occurs within the visit
            is_outcome_occurs_in_time_horizon: bool = (
                (
                    # ensure there is an outcome
                    # (needed in case there are 0 outcomes)
                    curr_outcome_idx
                    < len(outcome_times)
                )
                and (
                    # outcome occurs after visit starts
                    prediction_start
                    <= outcome_times[curr_outcome_idx]
                )
                and (
                    # outcome occurs before visit ends
                    outcome_times[curr_outcome_idx]
                    <= prediction_end
                )
            )
            # Assume no censoring for visits
            is_censored: bool = False

            if is_outcome_occurs_in_time_horizon:
                results.append(Label(time=prediction_start, value=True))
            elif not is_censored:
                # Not censored + no outcome => FALSE
                results.append(Label(time=prediction_start, value=False))

        return results

    def get_labeler_type(self) -> LabelType:
        return "boolean"


##########################################################
##########################################################
# Abstract classes derived from TimeHorizonEventLabeler
##########################################################
##########################################################


class CodeLabeler(TimeHorizonEventLabeler):
    """Apply a label based on 1+ outcome_codes' occurrence(s) over a fixed time horizon."""

    def __init__(
        self,
        outcome_codes: List[str],
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Optional[Callable] = None,
    ):
        """Create a CodeLabeler, which labels events whose index in your Ontology is in `self.outcome_codes`

        Args:
            prediction_codes (List[int]): Events that count as an occurrence of the outcome.
            time_horizon (TimeHorizon): An interval of time. If the event occurs during this time horizon, then
                the label is TRUE. Otherwise, FALSE.
            prediction_codes (Optional[List[int]]): If not None, limit events at which you make predictions to
                only events with an `event.code` in these codes.
            prediction_time_adjustment_func (Optional[Callable]). A function that takes in a `datetime.datetime`
                and returns a different `datetime.datetime`. Defaults to the identity function.
        """
        self.outcome_codes: List[str] = outcome_codes
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_codes: Optional[List[str]] = prediction_codes
        self.prediction_time_adjustment_func: Callable = (
            prediction_time_adjustment_func if prediction_time_adjustment_func else identity
        )

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return each event's start time (possibly modified by prediction_time_adjustment_func)
        as the time to make a prediction. Default to all events whose `code` is in `self.prediction_codes`."""
        times: List[datetime.datetime] = []
        last_time = None
        for e in patient.events:
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(e.start)
            if ((self.prediction_codes is None) or (e.code in self.prediction_codes)) and (
                last_time != prediction_time
            ):
                times.append(prediction_time)
                last_time = prediction_time
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.outcome_codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.outcome_codes:
                times.append(event.start)
        return times

    def allow_same_time_labels(self) -> bool:
        # We cannot allow labels at the same time as the codes since they will generally be available as features ...
        return False


class OMOPConceptCodeLabeler(CodeLabeler):
    """Same as CodeLabeler, but add the extra step of mapping OMOP concept IDs
    (stored in `omop_concept_ids`) to femr codes (stored in `codes`)."""

    # parent OMOP concept codes, from which all the outcome
    # are derived (as children from our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Optional[Callable] = None,
    ):
        outcome_codes: List[int] = list(
            get_femr_codes(
                ontology,
                self.original_omop_concept_codes,
                is_ontology_expansion=True,
            )
        )
        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
            prediction_time_adjustment_func=prediction_time_adjustment_func
            if prediction_time_adjustment_func
            else identity,
        )


if __name__ == "__main__":
    pass
