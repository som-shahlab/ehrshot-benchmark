"""Labeling functions for OMOP data based on lab values."""
import datetime
from abc import abstractmethod
from typing import Any, List, Optional, Set
from femr import Patient, Event
from femr.extension import datasets as extension_datasets
from ehrshot.labelers.core import Label, Labeler, LabelType
from ehrshot.labelers.omop import get_femr_codes

##########################################################
##########################################################
# Labelers based on Lab Values.
#
# The difference between these Labelers and the ones based on codes
# is that these Labelers are based on lab values, not coded
# diagnoses. Thus, they may catch more cases of a given
# condition due to under-coding, but they are also more
# likely to be noisy.
##########################################################
##########################################################


class InstantLabValueLabeler(Labeler):
    """Apply a multi-class label for the outcome of a lab test.

    Prediction Time: Immediately before lab result is returned (i.e. 1 minute before)
    Time Horizon: The next immediate result for this lab test
    Label: Severity level of lab

    Excludes:
        - Labels that occur at the same exact time as the very first event in a patient's history
    """

    # parent OMOP concept codes, from which all the outcomes are derived (as children in our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology = ontology
        self.outcome_codes: Set[str] = get_femr_codes(
            ontology,
            self.original_omop_concept_codes,
            is_ontology_expansion=True,
        )

    def label(self, patient: Patient, is_show_warnings: bool = False) -> List[Label]:
        labels: List[Label] = []
        for e in patient.events:
            if patient.events[0].start == e.start:
                # Ignore events that occur at the same time as the first event in the patient's history
                continue
            if e.code in self.outcome_codes:
                # This is an outcome event
                if e.value is not None:
                    try:
                        # `e.unit` is string of form "mg/dL", "ounces", etc.
                        label: int = self.label_to_int(self.value_to_label(str(e.value), str(e.unit)))
                        prediction_time: datetime.datetime = e.start - datetime.timedelta(minutes=1)
                        labels.append(Label(prediction_time, label))
                    except Exception as exception:
                        if is_show_warnings:
                            print(
                                f"Warning: Error parsing value='{e.value}' with unit='{e.unit}'"
                                f" for code='{e.code}' @ {e.start} for patient_id='{patient.patient_id}'"
                                f" | Exception: {exception}"
                            )
        return labels

    def get_labeler_type(self) -> LabelType:
        return "categorical"

    def label_to_int(self, label: str) -> int:
        if label == "normal":
            return 0
        elif label == "mild":
            return 1
        elif label == "moderate":
            return 2
        elif label == "severe":
            return 3
        raise ValueError(f"Invalid label without a corresponding int: {label}")

    @abstractmethod
    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        """Convert `value` to a string label: "mild", "moderate", "severe", or "normal".
        NOTE: Some units have the form 'mg/dL (See scan or EMR data for detail)', so you
        need to use `.startswith()` to check for the unit you want.
        """
        return "normal"