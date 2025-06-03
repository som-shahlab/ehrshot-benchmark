"""Labeling functions for OMOP data."""
import datetime
import warnings
from typing import List, Optional, Tuple, Any, Dict
import pandas as pd
from femr import Event, Patient
from femr.extension import datasets as extension_datasets
from ehrshot.labelers.core import Label, Labeler, LabelType, TimeHorizon, TimeHorizonEventLabeler, LabeledPatients
from ehrshot.labelers.omop import (
    CodeLabeler,
    WithinVisitLabeler,
    does_exist_event_within_time_range,
    get_death_concepts,
    get_femr_codes,
    get_icu_events,
    get_inpatient_admission_events,
    move_datetime_to_end_of_day,
)
from ehrshot.labelers.omop_inpatient_admissions import get_inpatient_admission_discharge_times
from ehrshot.labelers.omop_lab_values import InstantLabValueLabeler

##########################################################
##########################################################
# CLMBR Benchmark Tasks
# See: https://www.medrxiv.org/content/10.1101/2022.04.15.22273900v1
# details on how this was reproduced.
#
# Citation: Guo et al.
# "EHR foundation models improve robustness in the presence of temporal distribution shift"
# Scientific Reports. 2023.
##########################################################
##########################################################


class Guo_LongLOSLabeler(Labeler):
    """Long LOS prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of admission whether the patient stays in hospital for >=7 days.

    Excludes:
        - Visits where discharge occurs on the same day as admission
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.long_time: datetime.timedelta = datetime.timedelta(days=7)
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def label(self, patient: Patient) -> List[Label]:
        """Label all admissions with admission length >= `self.long_time`"""
        labels: List[Label] = []
        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            # If admission and discharge are on the same day, then ignore
            if admission_time.date() == discharge_time.date():
                continue
            is_long_admission: bool = (discharge_time - admission_time) >= self.long_time
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(admission_time)
            labels.append(Label(prediction_time, is_long_admission))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class Guo_30DayReadmissionLabeler(TimeHorizonEventLabeler):
    """30-day readmissions prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of disharge whether the patient will be readmitted within 30 days.

    Excludes:
        - Patients readmitted on same day as discharge
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.time_horizon: TimeHorizon = TimeHorizon(
            start=datetime.timedelta(minutes=1), end=datetime.timedelta(days=30)
        )
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of inpatient admissions."""
        times: List[datetime.datetime] = []
        for admission_time, __ in get_inpatient_admission_discharge_times(patient, self.ontology):
            times.append(admission_time)
        return times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return end of admission as prediction timm."""
        times: List[datetime.datetime] = []
        admission_times = set()
        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(discharge_time)
            # Ignore patients who are readmitted the same day they were discharged b/c of data leakage
            if prediction_time.replace(hour=0, minute=0, second=0, microsecond=0) in admission_times:
                continue
            times.append(prediction_time)
            admission_times.add(admission_time.replace(hour=0, minute=0, second=0, microsecond=0))
        times = sorted(list(set(times)))
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon


class Guo_ICUAdmissionLabeler(WithinVisitLabeler):
    """ICU admission prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of admission whether the patient will be admitted to the ICU during their admission.

    Excludes:
        - Patients transfered on same day as admission
        - Visits where discharge occurs on the same day as admission
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        super().__init__(
            ontology=ontology,
            visit_start_adjust_func=move_datetime_to_end_of_day,
            visit_end_adjust_func=None,
        )

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        # Return the start times of all ICU admissions -- this is our outcome
        return [e.start for e in get_icu_events(patient, self.ontology)]  # type: ignore

    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Return all inpatient visits where ICU transfer does not occur on the same day as admission."""
        # Get all inpatient visits -- each visit comprises a prediction (start, end) time horizon
        all_visits: List[Event] = get_inpatient_admission_events(patient, self.ontology)
        # Exclude visits where ICU admission occurs on the same day as admission
        icu_transfer_dates: List[datetime.datetime] = [
            x.replace(hour=0, minute=0, second=0, microsecond=0) for x in self.get_outcome_times(patient)
        ]
        valid_visits: List[Event] = []
        for visit in all_visits:
            # If admission and discharge are on the same day, then ignore
            if visit.start.date() == visit.end.date():
                continue
            # If ICU transfer occurs on the same day as admission, then ignore
            if visit.start.replace(hour=0, minute=0, second=0, microsecond=0) in icu_transfer_dates:
                continue
            valid_visits.append(visit)
        return valid_visits


##########################################################
##########################################################
# MIMIC-III Benchmark Tasks
# See: https://www.nature.com/articles/s41597-019-0103-9/figures/7 for
# details on how this was reproduced.
#
# Citation: Harutyunyan, H., Khachatrian, H., Kale, D.C. et al.
# Multitask learning and benchmarking with clinical time series data.
# Sci Data 6, 96 (2019). https://doi.org/10.1038/s41597-019-0103-9
##########################################################
##########################################################


class Harutyunyan_DecompensationLabeler(CodeLabeler):
    """Decompensation prediction task from Harutyunyan et al. 2019.

    Hourly binary prediction task on whether the patient dies in the next 24 hours.
    Make prediction every 60 minutes after ICU admission, starting at hour 4.

    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 4 hours
        - ICU admissions with no events
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        # Next 24 hours
        time_horizon = TimeHorizon(datetime.timedelta(hours=0), datetime.timedelta(hours=24))
        # Death events
        outcome_codes = list(get_femr_codes(ontology, get_death_concepts(), is_ontology_expansion=True))
        # Save ontology for `get_prediction_times()`
        self.ontology = ontology

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
        )

    def is_apply_censoring(self) -> bool:
        """Consider censored patients to be alive."""
        return False

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of every hour after every ICU visit, up until death occurs or end of visit.
        Note that this requires creating an artificial event for each hour since there will only be one true
        event per ICU admission, but we'll need to create many subevents (at each hour) within this event.
        Also note that these events may not align with :00 minutes if the ICU visit does not start exactly "on the hour".

        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 4 hours
            - ICU admissions with no events
        """
        times: List[datetime.datetime] = []
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True)  # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        death_times: List[datetime.datetime] = self.get_outcome_times(patient)
        earliest_death_time: datetime.datetime = min(death_times) if len(death_times) > 0 else datetime.datetime.max
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=4)
                and does_exist_event_within_time_range(patient, e.start, e.end, exclude_event_idxs=icu_event_idxs)
            ):
                # Record every hour after admission (i.e. every hour between `e.start` and `e.end`),
                # but only after 4 hours have passed (i.e. start at `e.start + 4 hours`)
                # and only until the visit ends (`e.end`) or a death event occurs (`earliest_death_time`)
                end_of_stay: datetime.datetime = min(e.end, earliest_death_time)
                event_time = e.start + datetime.timedelta(hours=4)
                while event_time < end_of_stay:
                    times.append(event_time)
                    event_time += datetime.timedelta(hours=1)
        return times


class Harutyunyan_MortalityLabeler(WithinVisitLabeler):
    """In-hospital mortality prediction task from Harutyunyan et al. 2019.
    Single binary prediction task of whether patient dies within ICU admission 48 hours after admission.
    Make prediction 48 hours into ICU admission.

    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 48 hours
        - ICU admissions with no events before 48 hours
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        visit_start_adjust_func = lambda x: x + datetime.timedelta(
            hours=48
        )  # Make prediction 48 hours into ICU admission
        visit_end_adjust_func = lambda x: x
        super().__init__(ontology, visit_start_adjust_func, visit_end_adjust_func)

    def is_apply_censoring(self) -> bool:
        """Consider censored patients to be alive."""
        return False

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome"""
        outcome_codes = list(get_femr_codes(self.ontology, get_death_concepts(), is_ontology_expansion=True))
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in outcome_codes:
                times.append(e.start)
        return times

    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Return a list of all ICU visits > 48 hours.

        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 48 hours
            - ICU admissions with no events before 48 hours
        """
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True)  # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        valid_events: List[Event] = []
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=48)
                and does_exist_event_within_time_range(
                    patient, e.start, e.start + datetime.timedelta(hours=48), exclude_event_idxs=icu_event_idxs
                )
            ):
                valid_events.append(e)
        return valid_events


class Harutyunyan_LengthOfStayLabeler(Labeler):
    """LOS remaining regression task from Harutyunyan et al. 2019.

    Hourly regression task on the patient's remaining length-of-stay (in hours) in the ICU.
    Make prediction every 60 minutes after ICU admission, starting at hour 4.

    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 4 hours
        - ICU admissions with no events
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology = ontology

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome"""
        outcome_codes = list(get_femr_codes(self.ontology, get_death_concepts(), is_ontology_expansion=True))
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in outcome_codes:
                times.append(e.start)
        return times

    def get_labeler_type(self) -> LabelType:
        return "numerical"

    def label(self, patient: Patient) -> List[Label]:
        """Return a list of Labels at every hour after every ICU visit, where each Label is the # of hours
        until the visit ends (or a death event occurs).
        Note that this requires creating an artificial event for each hour since there will only be one true
        event per ICU admission, but we'll need to create many subevents (at each hour) within this event.
        Also note that these events may not align with :00 minutes if the ICU visit does not start exactly "on the hour".

        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 4 hours
            - ICU admissions with no events
        """
        labels: List[Label] = []
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True)  # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        death_times: List[datetime.datetime] = self.get_outcome_times(patient)
        earliest_death_time: datetime.datetime = min(death_times) if len(death_times) > 0 else datetime.datetime.max
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=4)
                and does_exist_event_within_time_range(patient, e.start, e.end, exclude_event_idxs=icu_event_idxs)
            ):
                # Record every hour after admission (i.e. every hour between `e.start` and `e.end`),
                # but only after 4 hours have passed (i.e. start at `e.start + 4 hours`)
                # and only until the visit ends (`e.end`) or a death event occurs (`earliest_death_time`)
                end_of_stay: datetime.datetime = min(e.end, earliest_death_time)
                event_time = e.start + datetime.timedelta(hours=4)
                while event_time < end_of_stay:
                    los: float = (end_of_stay - event_time).total_seconds() / 3600
                    labels.append(Label(event_time, los))
                    event_time += datetime.timedelta(hours=1)
                    assert (
                        los >= 0
                    ), f"LOS should never be negative, but end_of_stay={end_of_stay} - event_time={event_time} = {end_of_stay - event_time} for patient {patient.patient_id}"
        return labels


##########################################################
##########################################################
# Abnormal Lab Value Tasks
#
# Citation: Few shot EHR benchmark (ours)
##########################################################
##########################################################


class ThrombocytopeniaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for thrombocytopenia based on platelet count (10^9/L).
    Thresholds: mild (<150), moderate(<100), severe(<50), and reference range."""

    original_omop_concept_codes = [
        "LOINC/LP393218-5",
        "LOINC/LG32892-8",
        "LOINC/777-3",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if value < 50:
            return "severe"
        elif value < 100:
            return "moderate"
        elif value < 150:
            return "mild"
        return "normal"


class HyperkalemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hyperkalemia using blood potassium concentration (mmol/L).
    Thresholds: mild(>5.5),moderate(>6),severe(>7), and abnormal range."""

    original_omop_concept_codes = [
        "LOINC/LG7931-1",
        "LOINC/LP386618-5",
        "LOINC/LG10990-6",
        "LOINC/6298-4",
        "LOINC/2823-3",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("mmol/l"):
                # mmol/L
                # Original OMOP concept ID: 8753
                value = value
            elif unit.startswith("meq/l"):
                # mEq/L (1-to-1 -> mmol/L)
                # Original OMOP concept ID: 9557
                value = value
            elif unit.startswith("mg/dl"):
                # mg / dL (divide by 18 to get mmol/L)
                # Original OMOP concept ID: 8840
                value = value / 18.0
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value > 7:
            return "severe"
        elif value > 6.0:
            return "moderate"
        elif value > 5.5:
            return "mild"
        return "normal"


class HypoglycemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hypoglycemia using blood glucose concentration (mmol/L).
    Thresholds: mild(<3), moderate(<3.5), severe(<=3.9), and abnormal range."""

    original_omop_concept_codes = [
        "SNOMED/33747003",
        "LOINC/LP416145-3",
        "LOINC/14749-6",
        # "LOINC/15074-8",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("mg/dl"):
                # mg / dL
                # Original OMOP concept ID: 8840, 9028
                value = value / 18
            elif unit.startswith("mmol/l"):
                # mmol / L (x 18 to get mg/dl)
                # Original OMOP concept ID: 8753
                value = value
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value < 3:
            return "severe"
        elif value < 3.5:
            return "moderate"
        elif value <= 3.9:
            return "mild"
        return "normal"


class HyponatremiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hyponatremia based on blood sodium concentration (mmol/L).
    Thresholds: mild (<=135),moderate(<130),severe(<125), and abnormal range."""

    original_omop_concept_codes = ["LOINC/LG11363-5", "LOINC/2951-2", "LOINC/2947-0"]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if value < 125:
            return "severe"
        elif value < 130:
            return "moderate"
        elif value <= 135:
            return "mild"
        return "normal"


class AnemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for anemia based on hemoglobin levels (g/L).
    Thresholds: mild(<120),moderate(<110),severe(<70), and reference range"""

    original_omop_concept_codes = [
        "LOINC/LP392452-1",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("g/dl"):
                # g / dL
                # Original OMOP concept ID: 8713
                # NOTE: This weird *10 / 100 is how Lawrence did it
                value = value * 10
            elif unit.startswith("mg/dl"):
                # mg / dL (divide by 1000 to get g/dL)
                # Original OMOP concept ID: 8840
                # NOTE: This weird *10 / 100 is how Lawrence did it
                value = value / 100
            elif unit.startswith("g/l"):
                value = value
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value < 70:
            return "severe"
        elif value < 110:
            return "moderate"
        elif value < 120:
            return "mild"
        return "normal"


##########################################################
##########################################################
# First Diagnosis Tasks
# See: https://github.com/som-shahlab/few_shot_ehr/tree/main
#
# Citation: Few shot EHR benchmark (ours)
##########################################################
##########################################################


class FirstDiagnosisTimeHorizonCodeLabeler(Labeler):
    """Predict if patient will have their *first* diagnosis of `self.root_concept_code` in the next (1, 365) days.

    Make prediction at 11:59pm on day of discharge from inpatient admission.

    Excludes:
        - Patients who have already had this diagnosis
    """

    root_concept_code = None  # OMOP concept code for outcome, e.g. "SNOMED/57054005"

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        assert (
            self.root_concept_code is not None
        ), "Must specify `root_concept_code` for `FirstDiagnosisTimeHorizonCodeLabeler`"
        self.ontology = ontology
        self.outcome_codes = list(get_femr_codes(ontology, [self.root_concept_code], is_ontology_expansion=True))
        self.time_horizon: TimeHorizon = TimeHorizon(datetime.timedelta(days=1), datetime.timedelta(days=365))

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return discharges that occur before first diagnosis of outcome as prediction times."""
        times: List[datetime.datetime] = []
        for __, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            prediction_time: datetime.datetime = move_datetime_to_end_of_day(discharge_time)
            times.append(prediction_time)
        times = sorted(list(set(times)))

        # Drop all times that occur after first diagnosis
        valid_times: List[datetime.datetime] = []
        outcome_times: List[datetime.datetime] = self.get_outcome_times(patient)
        if len(outcome_times) == 0:
            return times
        else:
            first_diagnosis_time: datetime.datetime = min(outcome_times)
            for t in times:
                if t <= first_diagnosis_time:
                    valid_times.append(t)
            return valid_times

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.outcome_codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.outcome_codes:
                times.append(event.start)
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def is_apply_censoring(self) -> bool:
        return False

    def allow_same_time_labels(self) -> bool:
        return False

    def get_labeler_type(self) -> LabelType:
        return "boolean"

    def label(self, patient: Patient) -> List[Label]:
        # NOTE: This is EXACTLY THE SAME as the `TimeHorizonLabeler`
        if len(patient.events) == 0:
            return []

        __, end_time = self.get_patient_start_end_times(patient)
        prediction_times: List[datetime.datetime] = self.get_prediction_times(patient)
        outcome_times: List[datetime.datetime] = self.get_outcome_times(patient)
        time_horizon: TimeHorizon = self.get_time_horizon()

        # Get (start, end) of time horizon. If end is None, then it's infinite (set timedelta to max)
        time_horizon_start: datetime.timedelta = time_horizon.start
        time_horizon_end: Optional[datetime.timedelta] = time_horizon.end  # `None` if infinite time horizon

        # For each prediction time, check if there is an outcome which occurs within the (start, end)
        # of the time horizon
        results: List[Label] = []
        curr_outcome_idx: int = 0
        last_time = None
        for time_idx, time in enumerate(prediction_times):
            if last_time is not None:
                assert (
                    time > last_time
                ), f"Must be ascending prediction times, instead got last_prediction_time={last_time} <= prediction_time={time} for patient {patient.patient_id} at curr_outcome_idx={curr_outcome_idx} | prediction_time_idx={time_idx} | start_prediction_time={prediction_times[0]}"

            last_time = time

            while curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] < time + time_horizon_start:
                # `curr_outcome_idx` is the idx in `outcome_times` that corresponds to the first
                # outcome EQUAL or AFTER the time horizon for this prediction time starts (if one exists)
                curr_outcome_idx += 1

            if curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] == time:
                if not self.allow_same_time_labels():
                    continue
                warnings.warn(
                    "You are making predictions at the same time as the target outcome."
                    "This frequently leads to label leakage."
                )

            # TRUE if an event occurs within the time horizon
            is_outcome_occurs_in_time_horizon: bool = (
                (
                    # ensure there is an outcome
                    # (needed in case there are 0 outcomes)
                    curr_outcome_idx
                    < len(outcome_times)
                )
                and (
                    # outcome occurs after time horizon starts
                    time + time_horizon_start
                    <= outcome_times[curr_outcome_idx]
                )
                and (
                    # outcome occurs before time horizon ends (if there is an end)
                    (time_horizon_end is None)
                    or outcome_times[curr_outcome_idx] <= time + time_horizon_end
                )
            )
            # TRUE if patient is censored (i.e. timeline ends BEFORE this time horizon ends,
            # so we don't know if the outcome happened after the patient timeline ends)
            # If infinite time horizon labeler, then assume no censoring
            is_censored: bool = end_time < time + time_horizon_end if (time_horizon_end is not None) else False

            if is_outcome_occurs_in_time_horizon:
                results.append(Label(time=time, value=True))
            elif not is_censored:
                # Not censored + no outcome => FALSE
                results.append(Label(time=time, value=False))
            else:
                if self.is_apply_censoring():
                    # Censored + no outcome => CENSORED
                    pass
                else:
                    # Censored + no outcome => FALSE
                    results.append(Label(time=time, value=False))

        return results


class PancreaticCancerCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 200684
    root_concept_code = "SNOMED/372003004"


class CeliacDiseaseCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 60270
    root_concept_code = "SNOMED/396331005"


class LupusCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 176684
    root_concept_code = "SNOMED/55464009"


class AcuteMyocardialInfarctionCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 21982
    root_concept_code = "SNOMED/57054005"


class CTEPHCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 1433
    root_concept_code = "SNOMED/233947005"


class EssentialHypertensionCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 4644483
    root_concept_code = "SNOMED/59621000"


class HyperlipidemiaCodeLabeler(FirstDiagnosisTimeHorizonCodeLabeler):
    # n = 3048320
    root_concept_code = "SNOMED/55822004"




##########################################################
##########################################################
# CheXpert
##########################################################
##########################################################

CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

def chexpert_apply_labeling_function(args: Tuple[Any, str, str, List[int], Optional[int]]) -> Dict[int, List[Label]]:
    """Apply a labeling function to the set of patients included in `patient_ids`.
    Gets called as a parallelized subprocess of the .apply() method of `Labeler`."""
    labeling_function: Any = args[0]
    path_to_chexpert_csv: str = args[1]
    path_to_patient_database: str = args[2]
    patient_ids: List[int] = args[3]
    num_labels: Optional[int] = args[4]

    chexpert_df = pd.read_csv(path_to_chexpert_csv, sep="\t")
    patients = PatientDatabase(path_to_patient_database)

    chexpert_df[CHEXPERT_LABELS] = (chexpert_df[CHEXPERT_LABELS] == 1) * 1

    patients_to_labels: Dict[int, List[Label]] = {}
    for patient_id in patient_ids:
        patient: Patient = patients[patient_id]  # type: ignore
        patient_df = chexpert_df[chexpert_df["patient_id"] == patient_id]

        if num_labels is not None and num_labels < len(patient_df):
            patient_df = patient_df.sample(n=num_labels, random_state=0)
        labels: List[Label] = labeling_function.label(patient, patient_df)
        patients_to_labels[patient_id] = labels

    return patients_to_labels


class ChexpertLabeler(Labeler):
    """CheXpert labeler.

    Multi-label classification task of patient's radiology reports.
    Make prediction 24 hours before radiology note is recorded.

    Excludes:
        - Radiology reports that are written <=24 hours of a patient's first event (i.e. `patient.events[0].start`)
    """

    def __init__(
        self,
        path_to_chexpert_csv: str,
    ):
        self.path_to_chexpert_csv = path_to_chexpert_csv

    def get_patient_start_end_times(self, patient: Patient) -> Tuple[datetime.datetime, datetime.datetime]:
        """Return the (start, end) of the patient timeline.

        Returns:
            Tuple[datetime.datetime, datetime.datetime]: (start, end)
        """
        return (patient.events[0].start, patient.events[-1].start)

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient has a radiology report"""

        chexpert_df = pd.read_csv(self.path_to_chexpert_csv, sep="\t")

        patient_df = chexpert_df.sort_values(by=["start"], ascending=True)

        start_time, _ = self.get_patient_start_end_times(patient)

        outcome_times = []
        for idx, row in patient_df.iterrows():
            label_time = row["start"]
            label_time = datetime.datetime.strptime(label_time, "%Y-%m-%d %H:%M:%S")
            prediction_time = label_time - timedelta(hours=24)

            if prediction_time <= start_time:
                continue
            outcome_times.append(label_time)

        return outcome_times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        outcome_times = self.get_outcome_times(patient)
        return [outcome_time - timedelta(hours=24) for outcome_time in outcome_times]

    def get_labeler_type(self) -> LabelType:
        return "categorical"

    def label(self, patient: Patient, patient_df: pd.DataFrame) -> List[Label]:
        labels: List[Label] = []

        patient_df = patient_df.sort_values(by=["start"], ascending=True)
        start_time, _ = self.get_patient_start_end_times(patient)

        for idx, row in patient_df.iterrows():
            label_time = row["start"]
            label_time = datetime.datetime.strptime(label_time, "%Y-%m-%d %H:%M:%S")
            prediction_time = label_time - timedelta(days=1)

            if prediction_time <= start_time:
                continue

            bool_labels = row[CHEXPERT_LABELS].astype(int).to_list()
            label_string = "".join([str(x) for x in bool_labels])
            label_num = int(label_string, 2)
            labels.append(Label(time=prediction_time, value=label_num))

        return labels

    def apply(
        self,
        path_to_patient_database: str,
        num_threads: int = 1,
        num_patients: Optional[int] = None,
        num_labels: Optional[int] = None,
    ) -> LabeledPatients:
        """Apply the `label()` function one-by-one to each Patient in a sequence of Patients.

        Args:
            path_to_patient_database (str, optional): Path to `PatientDatabase` on disk.
                Must be specified if `patients = None`
            num_threads (int, optional): Number of CPU threads to parallelize across. Defaults to 1.
            num_patients (Optional[int], optional): Number of patients to process - useful for debugging.
                If specified, will take the first `num_patients` in the provided `PatientDatabase` / `patients` list.
                If None, use all patients.

        Returns:
            LabeledPatients: Maps patients to labels
        """
        # Split patient IDs across parallelized processes
        chexpert_df = pd.read_csv(self.path_to_chexpert_csv, sep="\t")
        pids = list(chexpert_df["patient_id"].unique())

        if num_patients is not None:
            pids = pids[:num_patients]

        pid_parts = np.array_split(pids, num_threads)

        # Multiprocessing
        tasks = [
            (self, self.path_to_chexpert_csv, path_to_patient_database, pid_part, num_labels) for pid_part in pid_parts
        ]

        with multiprocessing.Pool(num_threads) as pool:
            results: List[Dict[int, List[Label]]] = list(pool.imap(chexpert_apply_labeling_function, tasks))

        # Join results and return
        patients_to_labels: Dict[int, List[Label]] = dict(collections.ChainMap(*results))
        return LabeledPatients(patients_to_labels, self.get_labeler_type())



if __name__ == "__main__":
    pass
