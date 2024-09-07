"""Core labeling functionality/schemas, shared across all labeling functions."""
from __future__ import annotations

import collections
import csv
import datetime
import hashlib
import multiprocessing
import pprint
import struct
import warnings
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
from nptyping import NDArray

from femr import Patient
from femr.datasets import PatientDatabase
from femr.extension import datasets as extension_datasets


@dataclass(frozen=True)
class TimeHorizon:
    """An interval of time. Mandatory `start`, optional `end`."""

    start: datetime.timedelta
    end: datetime.timedelta | None  # If NONE, then infinite time horizon


@dataclass(frozen=True)
class SurvivalValue:
    """Used for survival tasks."""

    time_to_event: datetime.timedelta
    is_censored: bool  # TRUE if this patient was censored


LabelType = Union[
    Literal["boolean"],
    Literal["numerical"],
    Literal["survival"],
    Literal["categorical"],
]

VALID_LABEL_TYPES = ["boolean", "numerical", "survival", "categorical"]


@dataclass
class Label:
    """An individual label for a particular patient at a particular time.
    The prediction for this label is made with all data <= time."""

    time: datetime.datetime
    value: Union[bool, int, float, SurvivalValue]


def _apply_labeling_function(
    args: Tuple[Labeler, Optional[Sequence[Patient]], Optional[str], List[int]]
) -> Dict[int, List[Label]]:
    """Apply a labeling function to the set of patients included in `patient_ids`.
    Gets called as a parallelized subprocess of the .apply() method of `Labeler`."""
    labeling_function: Labeler = args[0]
    patients: Optional[Sequence[Patient]] = args[1]
    path_to_patient_database: Optional[str] = args[2]
    patient_ids: List[int] = args[3]

    if path_to_patient_database is not None:
        patients = PatientDatabase(path_to_patient_database)

    # Hacky workaround for Ontology not being picklable
    if (
        hasattr(labeling_function, "ontology")  # type: ignore
        and labeling_function.ontology is None  # type: ignore
        and path_to_patient_database  # type: ignore
    ):  # type: ignore
        labeling_function.ontology = patients.get_ontology()  # type: ignore
    if (
        hasattr(labeling_function, "labeler")
        and hasattr(labeling_function.labeler, "ontology")
        and labeling_function.labeler.ontology is None
        and path_to_patient_database
    ):
        labeling_function.labeler.ontology = patients.get_ontology()  # type: ignore

    patients_to_labels: Dict[int, List[Label]] = {}
    for patient_id in patient_ids:
        patient: Patient = patients[patient_id]  # type: ignore
        labels: List[Label] = labeling_function.label(patient)
        patients_to_labels[patient_id] = labels

    return patients_to_labels


def load_labeled_patients(filename: str) -> LabeledPatients:
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) != 0, "Must have at least one label to load it"

        labeler_type: LabelType = cast(LabelType, rows[0]["label_type"])
        labels = collections.defaultdict(list)

        for row in rows:
            value: Union[bool, SurvivalValue, int, float]
            if labeler_type == "survival":
                value = SurvivalValue(
                    time_to_event=datetime.timedelta(minutes=float(row["value"])),
                    is_censored=row["is_censored"].lower() == "true",
                )
            elif labeler_type == "boolean":
                value = row["value"].lower() == "true"
            elif labeler_type == "categorical":
                value = int(row["value"])
            else:
                value = float(row["value"])

            time = datetime.datetime.fromisoformat(row["prediction_time"])
            if time.second != 0:
                # warnings.warn(
                #     "FEMR only supports minute level time resolution. "
                #     "Rounding down to nearest minute."
                # )
                time = time.replace(second=0)
            # assert time.second == 0, "FEMR only supports minute level time resolution"

            labels[int(row["patient_id"])].append(Label(time=time, value=value))

        return LabeledPatients(labels, labeler_type)


class LabeledPatients(MutableMapping[int, List[Label]]):
    """Maps patients to labels.

    Wrapper class around the output of an LF's `apply()` function
    """

    def __init__(
        self,
        patients_to_labels: Dict[int, List[Label]],
        labeler_type: LabelType,
    ):
        """Construct a `LabeledPatients` object from the output of an LF's `apply()` function.

        Args:
            patients_to_labels (Dict[int, List[Label]]): [key] = patient ID, [value] = labels for this patient
            labeler_type (LabelType): Type of labeler
        """
        self.patients_to_labels: Dict[int, List[Label]] = patients_to_labels
        self.labeler_type: LabelType = labeler_type

    def save(self, target_filename) -> None:
        with open(target_filename, "w") as f:
            writer = csv.writer(f)
            header = ["patient_id", "prediction_time", "label_type", "value"]
            if self.labeler_type == "survival":
                header.append("is_censored")
            writer.writerow(header)
            pids = sorted(list(self.patients_to_labels.keys()))
            for pid in pids:
                labels = sorted(self.patients_to_labels[pid], key=lambda x: (x.time, x.value))
                for label in labels:
                    if self.labeler_type == "survival":
                        assert isinstance(label.value, SurvivalValue)
                        writer.writerow(
                            [
                                patient,
                                label.time.isoformat(),
                                self.labeler_type,
                                label.value.time_to_event / datetime.timedelta(minutes=1),
                                label.value.is_censored,
                            ]
                        )
                    else:
                        writer.writerow([patient, label.time.isoformat(), self.labeler_type, label.value])

    def get_labels_from_patient_idx(self, idx: int) -> List[Label]:
        return self.patients_to_labels[idx]

    def get_all_patient_ids(self) -> List[int]:
        return sorted(list(self.patients_to_labels.keys()))

    def get_patients_to_labels(self) -> Dict[int, List[Label]]:
        return self.patients_to_labels

    def get_labeler_type(self) -> LabelType:
        return self.labeler_type

    def as_numpy_arrays(
        self,
    ) -> Tuple[
        NDArray[Literal["n_patients, 1"], np.int64],
        NDArray[Literal["n_patients, 1 or 2"], Any],
        NDArray[Literal["n_patients, 1"], np.datetime64],
    ]:
        """Convert `patients_to_labels` to a tuple of NDArray's.

        One NDArray for each of:
            Patient ID, Label value, Label time

        Returns:
            Tuple[NDArray, NDArray, NDArray]: (Patient IDs, Label values, Label time)
        """
        patient_ids: List[int] = []
        label_values: List[Any] = []
        label_times: List[datetime.datetime] = []
        if self.labeler_type in ["boolean", "numerical", "categorical"]:
            for patient_id, labels in self.patients_to_labels.items():
                for label in labels:
                    patient_ids.append(patient_id)
                    label_values.append(label.value)
                    label_times.append(label.time)
        elif self.labeler_type in ["survival"]:
            # If SurvivalValue labeler, then label value is a tuple of (time to event, is censored)
            for patient_id, labels in self.patients_to_labels.items():
                for label in labels:
                    survival_value: SurvivalValue = cast(SurvivalValue, label.value)
                    patient_ids.append(patient_id)
                    label_values.append(
                        [
                            survival_value.time_to_event,
                            survival_value.is_censored,
                        ]
                    )
                    label_times.append(label.time)
        else:
            raise ValueError("Other label types are not implemented yet for this method")
        return (
            np.array(patient_ids),
            np.array(label_values),
            np.array(label_times),
        )

    def get_num_patients(self, is_include_empty_labels: bool = False) -> int:
        """Return the total number of patients. Defaults to only patients with at least one label.
        If `is_include_empty_labels = True`, include patients with zero associated labels.
        """
        if is_include_empty_labels:
            return len(self)
        return len({key: val for key, val in self.get_patients_to_labels().items() if len(val) > 0})

    def get_patients_with_labels(self) -> List[int]:
        """Return the IDs of patients with at least one label."""
        patient_ids: List[int] = [key for key, val in self.get_patients_to_labels().items() if len(val) > 0]
        return patient_ids

    def get_patients_with_label_values(self, values: List[Any]) -> List[int]:
        """Return the IDs of patients with at least one label whose value is in `values`."""
        patient_ids: set = set()
        for patient, labels in self.items():
            for label in labels:
                # NOTE: you can't use `label.value in values` because `in` does an implicit type conversion,
                # thus `1.0 in [True]` will return True incorrectly
                for v in values:
                    if label.value == v and isinstance(label.value, type(v)):
                        patient_ids.add(patient)
                        break
        return list(patient_ids)

    def get_num_labels(self) -> int:
        """Return the total number of labels across all patients."""
        total: int = 0
        for labels in self.patients_to_labels.values():
            total += len(labels)
        return total

    def as_list_of_label_tuples(self) -> List[Tuple[int, Label]]:
        """Convert `patients_to_labels` to a list of (patient_id, Label) tuples."""
        result: List[Tuple[int, Label]] = []
        for patient_id, labels in self.patients_to_labels.items():
            for label in labels:
                result.append((int(patient_id), label))
        return result

    @classmethod
    def load_from_numpy(
        cls,
        patient_ids: NDArray[Literal["n_patients, 1"], np.int64],
        label_values: NDArray[Literal["n_patients, 1 or 2"], Any],
        label_times: NDArray[Literal["n_patients, 1"], datetime.datetime],
        labeler_type: LabelType,
    ) -> LabeledPatients:
        """Create a :class:`LabeledPatients` from NDArray labels.

            Inverse of `as_numpy_arrays()`

        Args:
            patient_ids (NDArray): Patient IDs for the corresponding label.
            label_values (NDArray): Values for the corresponding label.
            label_times (NDArray): Times that the corresponding label occurs.
            labeler_type (LabelType): LabelType of the corresponding labels.
        """
        patients_to_labels: DefaultDict[int, List[Label]] = collections.defaultdict(list)
        for patient_id, l_value, l_time in zip(patient_ids, label_values, label_times):
            if labeler_type in ["boolean", "numerical", "categorical"]:
                patients_to_labels[patient_id].append(Label(time=l_time, value=l_value))
            elif labeler_type in ["survival"]:
                patients_to_labels[patient_id].append(
                    Label(
                        time=l_time,
                        value=SurvivalValue(time_to_event=l_value[0], is_censored=l_value[1]),
                    )
                )
            else:
                raise ValueError("Other label types are not implemented yet for this method")
        return LabeledPatients(dict(patients_to_labels), labeler_type)

    def __str__(self):
        """Return string representation."""
        return "LabeledPatients:\n" + pprint.pformat(self.patients_to_labels)

    def __getitem__(self, key):
        """Necessary for implementing MutableMapping."""
        return self.patients_to_labels[key]

    def __setitem__(self, key, item):
        """Necessary for implementing MutableMapping."""
        self.patients_to_labels[key] = item

    def __delitem__(self, key):
        """Necessary for implementing MutableMapping."""
        del self.patients_to_labels[key]

    def __iter__(self):
        """Necessary for implementing MutableMapping."""
        return iter(self.patients_to_labels)

    def __len__(self):
        """Necessary for implementing MutableMapping."""
        return len(self.patients_to_labels)


class Labeler(ABC):
    """An interface for labeling functions.

    A labeling function applies a label to a specific datetime in a given patient's timeline.
    It can be thought of as generating the following list given a specific patient:
        [(patient ID, datetime_1, label_1), (patient ID, datetime_2, label_2), ... ]
    Usage:
    ```
        labeling_function: Labeler = Labeler(...)
        patients: Sequence[Patient] = ...
        labels: LabeledPatient = labeling_function.apply(patients)
    ```
    """

    @abstractmethod
    def label(self, patient: Patient) -> List[Label]:
        """Apply every label that is applicable to the provided patient.

        This is only called once per patient.

        Args:
            patient (Patient): A patient object

        Returns:
            List[Label]: A list of :class:`Label` containing every label for the given patient
        """
        pass

    def get_patient_start_end_times(self, patient: Patient) -> Tuple[datetime.datetime, datetime.datetime]:
        """Return the (start, end) of the patient timeline.

        Returns:
            Tuple[datetime.datetime, datetime.datetime]: (start, end)
        """
        return (patient.events[0].start, patient.events[-1].start)

    @abstractmethod
    def get_labeler_type(self) -> LabelType:
        """Return what type of labels this labeler returns. See the Label class."""
        pass

    def apply(
        self,
        path_to_patient_database: Optional[str] = None,
        patients: Optional[Sequence[Patient]] = None,
        num_threads: int = 1,
        num_patients: Optional[int] = None,
    ) -> LabeledPatients:
        """Apply the `label()` function one-by-one to each Patient in a sequence of Patients.

        Args:
            path_to_patient_database (str, optional): Path to `PatientDatabase` on disk.
                Must be specified if `patients = None`
            patients (Sequence[Patient], optional): An Sequence (i.e. list) of `Patient` objects.
                Must be specified if `path_to_patient_database = None`
                Typically this will be a `PatientDatabase` object.
            num_threads (int, optional): Number of CPU threads to parallelize across. Defaults to 1.
            num_patients (Optional[int], optional): Number of patients to process - useful for debugging.
                If specified, will take the first `num_patients` in the provided `PatientDatabase` / `patients` list.
                If None, use all patients.

        Returns:
            LabeledPatients: Maps patients to labels
        """
        if (patients is None and path_to_patient_database is None) or (
            patients is not None and path_to_patient_database is not None
        ):
            raise ValueError("Must specify exactly one of `patient_database` or `path_to_patient_database`")

        if path_to_patient_database:
            # Load patientdatabase if specified
            assert patients is None
            patient_database = PatientDatabase(path_to_patient_database)
            num_patients = len(patient_database) if not num_patients else num_patients
            pids = list(patient_database)
        else:
            # Use `patients` if specified
            assert patients is not None
            num_patients = len(patients) if not num_patients else num_patients
            pids = [p.patient_id for p in patients[:num_patients]]

        # Split patient IDs across parallelized processes
        pid_parts = np.array_split(pids, num_threads * 10)

        # NOTE: Super hacky workaround to pickling limitations
        if hasattr(self, "ontology") and isinstance(self.ontology, extension_datasets.Ontology):  # type: ignore
            # Remove ontology due to pickling, add it back later
            self.ontology: extension_datasets.Ontology = None  # type: ignore
        if (
            hasattr(self, "labeler")
            and hasattr(self.labeler, "ontology")
            and isinstance(self.labeler.ontology, extension_datasets.Ontology)
        ):
            # If NLabelsPerPatient wrapper, go to sublabeler and remove ontology due to pickling
            self.labeler.ontology: extension_datasets.Ontology = None  # type: ignore

        # Multiprocessing
        tasks = [(self, patients, path_to_patient_database, pid_part) for pid_part in pid_parts if len(pid_part) > 0]

        if num_threads != 1:
            ctx = multiprocessing.get_context("forkserver")
            with ctx.Pool(num_threads) as pool:
                results = []
                for res in pool.imap_unordered(_apply_labeling_function, tasks):
                    results.append(res)
        else:
            results = []
            for task in tasks:
                results.append(_apply_labeling_function(task))

        # Join results and return
        patients_to_labels: Dict[int, List[Label]] = dict(collections.ChainMap(*results))
        return LabeledPatients(patients_to_labels, self.get_labeler_type())


##########################################################
# Specific Labeler Superclasses
##########################################################


class TimeHorizonEventLabeler(Labeler):
    """Label events that occur within a particular time horizon.
    This support both "finite" and "infinite" time horizons.

    The time horizon can be "fixed" (i.e. has both a start and end date), or "infinite" (i.e. only a start date)

    A TimeHorizonEventLabeler enables you to label events that occur within a particular
    time horizon (i.e. `TimeHorizon`). It is a boolean event that is TRUE if the event of interest
    occurs within that time horizon, and FALSE if it doesn't occur by the end of the time horizon.

    No labels are generated if the patient record is "censored" before the end of the horizon and `is_apply_censoring = True`.
    Note that this defaults to `is_apply_censoring = True`.

    You are required to implement three methods:
        get_outcome_times() for defining the datetimes of the event of interset
        get_prediction_times() for defining the datetimes at which we make our predictions
        get_time_horizon() for defining the length of time (i.e. `TimeHorizon`) to use for the time horizon
    """

    @abstractmethod
    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a sorted list containing the datetimes that the event of interest "occurs".

        IMPORTANT: Must be sorted ascending (i.e. start -> end of timeline)

        Args:
            patient (Patient): A patient object

        Returns:
            List[datetime.datetime]: A list of datetimes, one corresponding to an occurrence of the outcome
        """
        pass

    @abstractmethod
    def get_time_horizon(self) -> TimeHorizon:
        """Return time horizon for making predictions with this labeling function.

        Return the (start offset, end offset) of the time horizon (from the prediction time)
        used for labeling whether an outcome occurred or not. These can be arbitrary timedeltas.

        If end offset is None, then the time horizon is infinite (i.e. only has a start offset).
        If end offset is not None, then the time horizon is finite (i.e. has both a start and end offset),
            and it must be true that end offset >= start offset.

        Example:
            X is the time that you're making a prediction (given by `get_prediction_times()`)
            (A,B) is your time horizon (given by `get_time_horizon()`)
            O is an outcome (given by `get_outcome_times()`)

            Then given a patient timeline:
                X-----(X+A)------(X+B)------


            This has a label of TRUE:
                X-----(X+A)--O---(X+B)------

            This has a label of TRUE:
                X-----(X+A)--O---(X+B)----O-

            This has a label of FALSE:
                X---O-(X+A)------(X+B)------

            This has a label of FALSE:
                X-----(X+A)------(X+B)--O---
        """
        pass

    @abstractmethod
    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a sorted list containing the datetimes at which we'll make a prediction.

        IMPORTANT: Must be sorted ascending (i.e. start -> end of timeline)
        """
        pass

    def get_patient_start_end_times(self, patient: Patient) -> Tuple[datetime.datetime, datetime.datetime]:
        """Return the datetimes that we consider the (start, end) of this patient."""
        return (patient.events[0].start, patient.events[-1].start)

    def get_labeler_type(self) -> LabelType:
        """Return boolean labels (TRUE if event occurs in TimeHorizon, FALSE otherwise)."""
        return "boolean"

    def allow_same_time_labels(self) -> bool:
        """Whether or not to allow labels with events at the same time as prediction"""
        return True

    def is_apply_censoring(self) -> bool:
        """If TRUE, then a censored patient with no outcome -> IGNORED.
        If FALSE, then a censored patient with no outcome -> FALSE.
        """
        return True

    def label(self, patient: Patient) -> List[Label]:
        """Return a list of Labels for an individual patient.

        Assumes that events in `patient.events` are already sorted in chronologically
        ascending order (i.e. start -> end).

        Args:
            patient (Patient): A patient object

        Returns:
            List[Label]: A list containing a label for each datetime returned by `get_prediction_times()`
        """
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


class NLabelsPerPatientLabeler(Labeler):
    """Restricts `self.labeler` to returning a max of `self.k` labels per patient."""

    def __init__(self, labeler: Labeler, num_labels: int = 1, seed: int = 1):
        self.labeler: Labeler = labeler
        self.num_labels: int = num_labels  # number of labels per patient
        self.seed: int = seed

    def label(self, patient: Patient) -> List[Label]:
        labels: List[Label] = self.labeler.label(patient)
        if len(labels) <= self.num_labels:
            return labels
        hash_to_label_list: List[Tuple[int, int, Label]] = [
            (i, compute_random_num(self.seed, patient.patient_id, i), labels[i]) for i in range(len(labels))
        ]
        hash_to_label_list.sort(key=lambda a: a[1])
        n_hash_to_label_list: List[Tuple[int, int, Label]] = hash_to_label_list[: self.num_labels]
        n_hash_to_label_list.sort(key=lambda a: a[0])
        n_labels: List[Label] = [hash_to_label[2] for hash_to_label in n_hash_to_label_list]
        return n_labels

    def get_labeler_type(self) -> LabelType:
        return self.labeler.get_labeler_type()


def compute_random_num(seed: int, num_1: int, num_2: int):
    network_num_1 = struct.pack("!q", num_1)
    network_num_2 = struct.pack("!q", num_2)
    network_seed = struct.pack("!q", seed)

    to_hash = network_seed + network_num_1 + network_num_2

    hash_object = hashlib.sha256()
    hash_object.update(to_hash)
    hash_value = hash_object.digest()

    result = 0
    for i in range(len(hash_value)):
        result = (result * 256 + hash_value[i]) % 100

    return result
