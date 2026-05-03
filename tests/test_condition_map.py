"""
tests/test_condition_map.py
============================
Tests for condition mapping file column resolution.

Verifies that when a condition_map_file is used, BD_ column names in the
data file are correctly matched against the mapping regardless of whether
the data uses Gold-style (CPRD_FOO_BAR) or Aurum-style (CPRDAURUM_FOOBAR)
naming conventions.

Regression test for: mapping file with GOLD="CPRD_ACTINIC_KERATOSIS" failing
to match data column "BD_MEDI:CPRDAURUM_ACTINICKERATOSIS:2", causing the
pipeline to fall back to analysing all 286 columns.
"""
import csv
import os
import tempfile
import pytest

from main.preprocessing import _norm_bd_frag, _load_condition_map


class TestNormBdFrag:
    """Unit tests for the normalisation helper."""

    def test_gold_format(self):
        assert _norm_bd_frag("BD_MEDI:CPRD_ACTINIC_KERATOSIS") == "ACTINICKERATOSIS"

    def test_aurum_format(self):
        assert _norm_bd_frag("BD_MEDI:CPRDAURUM_ACTINICKERATOSIS") == "ACTINICKERATOSIS"

    def test_aurum_format_with_suffix(self):
        # Suffix stripped before calling _norm_bd_frag (as preprocessing does)
        from re import sub
        stripped = sub(r":\d+$", "", "BD_MEDI:CPRDAURUM_ACTINICKERATOSIS:2")
        assert _norm_bd_frag(stripped) == "ACTINICKERATOSIS"

    def test_gold_fragment_no_prefix(self):
        # Fragment from mapping file: no BD_MEDI: prefix
        assert _norm_bd_frag("CPRD_ACTINIC_KERATOSIS") == "ACTINICKERATOSIS"

    def test_aurum_fragment_no_prefix(self):
        assert _norm_bd_frag("CPRDAURUM_ACTINICKERATOSIS") == "ACTINICKERATOSIS"

    def test_gold_and_aurum_are_equal(self):
        from re import sub
        gold = _norm_bd_frag("CPRD_ACTINIC_KERATOSIS")
        aurum = _norm_bd_frag(sub(r":\d+$", "", "BD_MEDI:CPRDAURUM_ACTINICKERATOSIS:2"))
        assert gold == aurum

    def test_different_conditions_not_equal(self):
        assert _norm_bd_frag("CPRD_ACTINIC_KERATOSIS") != _norm_bd_frag("CPRD_DIABETES")


class TestLoadConditionMap:
    def test_basic_load(self, tmp_path):
        path = tmp_path / "map.csv"
        path.write_text(
            "Paper Short Name,Gold,Aurum\n"
            "act_keratosis,CPRD_ACTINIC_KERATOSIS,CPRDAURUM_ACTINICKERATOSIS\n"
        )
        result = _load_condition_map(str(path))
        assert "act_keratosis" in result
        assert result["act_keratosis"]["gold"] == "CPRD_ACTINIC_KERATOSIS"
        assert result["act_keratosis"]["aurum"] == "CPRDAURUM_ACTINICKERATOSIS"

    def test_empty_aurum(self, tmp_path):
        path = tmp_path / "map.csv"
        path.write_text(
            "Paper Short Name,Gold,Aurum\n"
            "act_keratosis,CPRD_ACTINIC_KERATOSIS,\n"
        )
        result = _load_condition_map(str(path))
        assert result["act_keratosis"]["gold"] == "CPRD_ACTINIC_KERATOSIS"
        assert result["act_keratosis"]["aurum"] == ""

    def test_strips_numeric_suffix(self, tmp_path):
        path = tmp_path / "map.csv"
        path.write_text(
            "Paper Short Name,Gold,Aurum\n"
            "cond_a,CPRD_FOO:1,CPRDAURUM_FOO:2\n"
        )
        result = _load_condition_map(str(path))
        assert result["cond_a"]["gold"] == "CPRD_FOO"
        assert result["cond_a"]["aurum"] == "CPRDAURUM_FOO"


class TestColumnFilterNormalisation:
    """Integration-style test for the column selection path.

    Simulates the filter-building and column-matching logic from preprocessing()
    to confirm that an Aurum-format data column is selected when the mapping
    file only provides a Gold fragment, and vice-versa.
    """

    def _build_filter_and_match(self, gold_frag, aurum_frag, candidate_cols):
        """Replicate the filter-building + matching logic from preprocessing()."""
        from re import sub

        col_filter_set: set = set()
        if gold_frag:
            col_filter_set.add(f"BD_MEDI:{gold_frag}")
        if aurum_frag:
            col_filter_set.add(f"BD_MEDI:{aurum_frag}")

        norm_filter = {_norm_bd_frag(f) for f in col_filter_set}
        return [
            c for c in candidate_cols
            if _norm_bd_frag(sub(r":\d+$", "", c)) in norm_filter
        ]

    def test_gold_fragment_matches_aurum_column(self):
        """GOLD='CPRD_ACTINIC_KERATOSIS' should select 'BD_MEDI:CPRDAURUM_ACTINICKERATOSIS:2'."""
        selected = self._build_filter_and_match(
            gold_frag="CPRD_ACTINIC_KERATOSIS",
            aurum_frag="",
            candidate_cols=[
                "BD_MEDI:CPRDAURUM_ACTINICKERATOSIS:2",
                "BD_MEDI:CPRDAURUM_DIABETES:1",
                "BD_MEDI:CPRD_OTHER_COND:3",
            ],
        )
        assert selected == ["BD_MEDI:CPRDAURUM_ACTINICKERATOSIS:2"]

    def test_aurum_fragment_matches_gold_column(self):
        """Aurum fragment should select a Gold-format data column."""
        selected = self._build_filter_and_match(
            gold_frag="",
            aurum_frag="CPRDAURUM_ACTINICKERATOSIS",
            candidate_cols=[
                "BD_MEDI:CPRD_ACTINIC_KERATOSIS:1",
                "BD_MEDI:CPRDAURUM_DIABETES:1",
            ],
        )
        assert selected == ["BD_MEDI:CPRD_ACTINIC_KERATOSIS:1"]

    def test_both_fragments_match_both_columns(self):
        """When both Gold and Aurum fragments are set, both source columns are selected."""
        selected = self._build_filter_and_match(
            gold_frag="CPRD_ACTINIC_KERATOSIS",
            aurum_frag="CPRDAURUM_ACTINICKERATOSIS",
            candidate_cols=[
                "BD_MEDI:CPRD_ACTINIC_KERATOSIS:1",
                "BD_MEDI:CPRDAURUM_ACTINICKERATOSIS:2",
                "BD_MEDI:CPRDAURUM_DIABETES:1",
            ],
        )
        assert set(selected) == {
            "BD_MEDI:CPRD_ACTINIC_KERATOSIS:1",
            "BD_MEDI:CPRDAURUM_ACTINICKERATOSIS:2",
        }

    def test_unrelated_columns_not_selected(self):
        selected = self._build_filter_and_match(
            gold_frag="CPRD_ACTINIC_KERATOSIS",
            aurum_frag="",
            candidate_cols=[
                "BD_MEDI:CPRDAURUM_DIABETES:1",
                "BD_MEDI:CPRD_ECZEMA:2",
            ],
        )
        assert selected == []
