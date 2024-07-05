from __future__ import annotations

import collections
import datetime
import functools
import random
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

# Following original femr.featurizers
from femr import Event, Patient
from femr.extension import datasets as extension_datasets
from femr.labelers import Label
from femr.featurizers.core import ColumnValue, Featurizer
from femr.featurizers.utils import OnlineStatistics

# Additional imports
from femr.featurizers.featurizers import _reshuffle_count_time_bins, exclusion_helper, ReservoirSampler
# TODO: Remove count featurizer
from femr.featurizers.featurizers import CountFeaturizer

class TextFeaturizer(Featurizer):
    """
    Produces one column per each diagnosis code, procedure code, and prescription code.
    The value in each column is the count of how many times that code appears in the patient record
    before the corresponding label.
    TODO: Add what is actually generated based on the data that we will find
    """

    def __init__(
        self,
        is_ontology_expansion: bool = False,
        excluded_codes: Iterable[str] = [],
        excluded_event_filter: Optional[Callable[[Event], bool]] = None,
        time_bins: Optional[List[datetime.timedelta]] = None,
        numeric_value_decile: bool = False,
        string_value_combination: bool = False,
        characters_for_string_values: int = 100,
    ):
        """
        Args:
            is_ontology_expansion (bool, optional): If TRUE, then do ontology expansion when counting codes.

                Example:
                    If `is_ontology_expansion=True` and your ontology is:
                        Code A -> Code B -> Code C
                    Where "->" denotes "is a parent of" relationship (i.e. A is a parent of B, B is a parent of C).
                    Then if we see 2 occurrences of Code "C", we count 2 occurrences of Code "B" and Code "A".

            excluded_codes (List[str], optional): A list of femr codes that we will ignore. Defaults to [].

            time_bins (Optional[List[datetime.timedelta]], optional): Group counts into buckets.
                Starts from the label time, and works backwards according to each successive value in `time_bins`.

                These timedeltas should be positive values, and will be internally converted to negative values

                If last value is `None`, then the last bucket will be from the penultimate value in `time_bins` to the
                    start of the patient's first event.

                Examples:
                    `time_bins = [
                        datetime.timedelta(days=90),
                        datetime.timedelta(days=180)
                    ]`
                        will create the following buckets:
                            [label time, -90 days], [-90 days, -180 days];
                    `time_bins = [
                        datetime.timedelta(days=90),
                        datetime.timedelta(days=180),
                        datetime.timedelta(years=100)
                    ]`
                        will create the following buckets:
                            [label time, -90 days], [-90 days, -180 days], [-180 days, -100 years];]
        """
        self.is_ontology_expansion: bool = is_ontology_expansion
        self.excluded_event_filter = functools.partial(
            exclusion_helper, fallback_function=excluded_event_filter, excluded_codes_set=set(excluded_codes)
        )
        self.time_bins: Optional[List[datetime.timedelta]] = time_bins
        self.characters_for_string_values: int = characters_for_string_values

        self.numeric_value_decile = numeric_value_decile
        self.string_value_combination = string_value_combination

        if self.time_bins is not None:
            assert len(set(self.time_bins)) == len(
                self.time_bins
            ), f"You cannot have duplicate values in the `time_bins` argument. You passed in: {self.time_bins}"

        self.observed_codes: Set[str] = set()
        self.observed_string_value: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        self.observed_numeric_value: Dict[str, ReservoirSampler] = collections.defaultdict(
            functools.partial(ReservoirSampler, 10000, 100)
        )

        self.finalized = False

    def get_codes(self, code: str, ontology: extension_datasets.Ontology) -> Iterator[str]:
        if self.is_ontology_expansion:
            for subcode in ontology.get_all_parents(code):
                yield subcode
        else:
            yield code

    def get_columns(self, event, ontology: extension_datasets.Ontology) -> Iterator[int]:
        if event.value is None:
            for code in self.get_codes(event.code, ontology):
                # If we haven't seen this code before, then add it to our list of included codes
                if code in self.code_to_column_index:
                    yield self.code_to_column_index[code]
        elif type(event.value) is str:
            k = (event.code, event.value[: self.characters_for_string_values])
            if k in self.code_string_to_column_index:
                yield self.code_string_to_column_index[k]
        else:
            if event.code in self.code_value_to_column_index:
                column, quantiles = self.code_value_to_column_index[event.code]
                for i, (start, end) in enumerate(zip(quantiles, quantiles[1:])):
                    if start <= event.value < end:
                        yield i + column

    def preprocess(self, patient: Patient, labels: List[Label], ontology: extension_datasets.Ontology):
        """Add every event code in this patient's timeline to `codes`."""
        for event in patient.events:
            # Check for excluded events
            if self.excluded_event_filter is not None and self.excluded_event_filter(event):
                continue

            if event.value is None:
                for code in self.get_codes(event.code, ontology):
                    # If we haven't seen this code before, then add it to our list of included codes
                    self.observed_codes.add(code)
            elif type(event.value) is str:
                if self.string_value_combination:
                    self.observed_string_value[(event.code, event.value[: self.characters_for_string_values])] += 1
            else:
                if self.numeric_value_decile:
                    self.observed_numeric_value[event.code].add(event.value)

    @classmethod
    def aggregate_preprocessed_featurizers(  # type: ignore[override]
        cls, featurizers: List[CountFeaturizer]
    ) -> CountFeaturizer:
        """After preprocessing a CountFeaturizer using multiprocessing (resulting in the list of featurizers
        contained in `featurizers`), this method aggregates all those featurizers into one CountFeaturizer.

        We need to collect all the unique event codes identified by each featurizer, and then create a new
        featurizer that combines all these codes
        """
        if len(featurizers) == 0:
            raise ValueError("You must pass in at least one featurizer to `aggregate_preprocessed_featurizers`")

        template_featurizer: CountFeaturizer = featurizers[0]

        for featurizer in featurizers[1:]:
            template_featurizer.observed_codes |= featurizer.observed_codes
            for k1, v1 in template_featurizer.observed_string_value.items():
                featurizer.observed_string_value[k1] += v1
            for k2, v2 in template_featurizer.observed_numeric_value.items():
                featurizer.observed_numeric_value[k2].values += v2.values

        return template_featurizer

    def finalize(self):
        if self.finalized:
            return

        self.finalized = True
        self.code_to_column_index = {}
        self.code_string_to_column_index = {}
        self.code_value_to_column_index = {}

        self.num_columns = 0

        for code in sorted(list(self.observed_codes)):
            self.code_to_column_index[code] = self.num_columns
            self.num_columns += 1

        for (code, val), count in sorted(list(self.observed_string_value.items())):
            if count > 1:
                self.code_string_to_column_index[(code, val)] = self.num_columns
                self.num_columns += 1

        for code, values in sorted(list(self.observed_numeric_value.items())):
            quantiles = sorted(list(set(np.quantile(values.values, np.linspace(0, 1, num=11)[1:-1]))))
            quantiles = [float("-inf")] + quantiles + [float("inf")]
            self.code_value_to_column_index[code] = (self.num_columns, quantiles)
            self.num_columns += len(quantiles) - 1

    def get_num_columns(self) -> int:
        self.finalize()

        if self.time_bins is None:
            return self.num_columns
        else:
            return self.num_columns * len(self.time_bins)

    def featurize(
        self,
        patient: Patient,
        labels: List[Label],
        ontology: Optional[extension_datasets.Ontology],
    ) -> List[List[ColumnValue]]:
        self.finalize()
        if ontology is None:
            raise ValueError("`ontology` can't be `None` for CountFeaturizer")

        all_columns: List[List[ColumnValue]] = []

        if self.time_bins is None:
            # Count the number of times each code appears in the patient's timeline
            # [key] = column idx
            # [value] = count of occurrences of events with that code (up to the label at `label_idx`)
            code_counter: Dict[int, int] = defaultdict(int)

            label_idx = 0
            for event in patient.events:
                if self.excluded_event_filter is not None and self.excluded_event_filter(event):
                    continue

                while event.start > labels[label_idx].time:
                    label_idx += 1
                    # Create all features for label at index `label_idx`
                    all_columns.append([ColumnValue(code, count) for code, count in code_counter.items()])
                    if label_idx >= len(labels):
                        # We've reached the end of the labels for this patient,
                        # so no point in continuing to count events past this point.
                        # Instead, we just return the counts of all events up to this point.
                        return all_columns

                for column_idx in self.get_columns(event, ontology):
                    code_counter[column_idx] += 1

            # For all labels that occur past the last event, add all
            # events' total counts as these labels' feature values (basically,
            # the featurization of these labels is the count of every single event)
            for _ in labels[label_idx:]:
                all_columns.append([ColumnValue(code, count) for code, count in code_counter.items()])

        else:
            # First, sort time bins in ascending order (i.e. [100 days, 90 days, 1 days] -> [1, 90, 100])
            time_bins: List[datetime.timedelta] = sorted([x for x in self.time_bins if x is not None])

            codes_per_bin: Dict[int, Deque[Tuple[int, datetime.datetime]]] = {
                i: deque() for i in range(len(self.time_bins) + 1)
            }

            code_counts_per_bin: Dict[int, Dict[int, int]] = {
                i: defaultdict(int) for i in range(len(self.time_bins) + 1)
            }

            label_idx = 0
            for event in patient.events:
                if self.excluded_event_filter is not None and self.excluded_event_filter(event):
                    continue
                while event.start > labels[label_idx].time:
                    _reshuffle_count_time_bins(
                        time_bins,
                        codes_per_bin,
                        code_counts_per_bin,
                        labels[label_idx],
                    )
                    label_idx += 1
                    # Create all features for label at index `label_idx`
                    all_columns.append(
                        [
                            ColumnValue(
                                code + i * self.num_columns,
                                count,
                            )
                            for i in range(len(self.time_bins))
                            for code, count in code_counts_per_bin[i].items()
                        ]
                    )

                    if label_idx >= len(labels):
                        # We've reached the end of the labels for this patient,
                        # so no point in continuing to count events past this point.
                        # Instead, we just return the counts of all events up to this point.
                        return all_columns

                for column_idx in self.get_columns(event, ontology):
                    codes_per_bin[0].append((column_idx, event.start))
                    code_counts_per_bin[0][column_idx] += 1

            for label in labels[label_idx:]:
                _reshuffle_count_time_bins(
                    time_bins,
                    codes_per_bin,
                    code_counts_per_bin,
                    label,
                )
                all_columns.append(
                    [
                        ColumnValue(
                            code + i * self.num_columns,
                            count,
                        )
                        for i in range(len(self.time_bins))
                        for code, count in code_counts_per_bin[i].items()
                    ]
                )

        return all_columns

    def is_needs_preprocessing(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"CountFeaturizer(number of included codes={self.num_columns})"

    def get_column_name(self, column_idx: int) -> str:
        def helper(actual_idx):
            for code, idx in self.code_to_column_index.items():
                if idx == actual_idx:
                    return code
            for (code, val), idx in self.code_string_to_column_index.items():
                if idx == actual_idx:
                    return f"{code} {val}"

            for code, (idx, quantiles) in self.code_value_to_column_index.items():
                offset = actual_idx - idx
                if 0 <= offset < len(quantiles) - 1:
                    return f"{code} [{quantiles[offset]}, {quantiles[offset+1]})"

            raise RuntimeError("Could not find name for " + str(actual_idx))

        if self.time_bins is None:
            return helper(column_idx)
        else:
            return helper(column_idx % self.num_columns) + f"_{self.time_bins[column_idx // self.num_columns]}"
