#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the skill normalizer module."""

import unittest

from ai_skills.skill_normalizer import (
    normalize_hardskills,
    normalize_softskills,
)


class TestNormalizeHardskills(unittest.TestCase):
    """Tests for hardskill normalization."""

    def test_empty_list(self):
        """Empty input returns empty string."""
        self.assertEqual(normalize_hardskills([]), "")

    def test_basic_normalization(self):
        """Skills are lowercased, deduplicated, and sorted."""
        result = normalize_hardskills(["Python", "PYTHON", "python"])
        self.assertEqual(result, "python")

    def test_alphabetical_sorting(self):
        """Skills are sorted alphabetically."""
        result = normalize_hardskills(["zsh", "bash", "aws"])
        self.assertEqual(result, "aws, bash, zsh")

    def test_canonicalization_rest(self):
        """REST variants are canonicalized."""
        result = normalize_hardskills(["RESTful", "restlets", "RESTLET"])
        self.assertEqual(result, "rest")

    def test_canonicalization_api(self):
        """API variants are canonicalized."""
        result = normalize_hardskills(["APIs", "Web API", "api"])
        self.assertEqual(result, "api")

    def test_canonicalization_dotnet(self):
        """All .NET variants become 'dotnet'."""
        result = normalize_hardskills([".NET", ".NET Core", "ASP.NET"])
        self.assertEqual(result, "dotnet")

    def test_canonicalization_cloud(self):
        """Cloud provider names are canonicalized."""
        result = normalize_hardskills(["Amazon Web Services", "Google Cloud Platform"])
        self.assertEqual(result, "aws, gcp")

    def test_mixed_skills(self):
        """Mixed skills are properly normalized."""
        result = normalize_hardskills([
            "Python", "JavaScript", ".NET Core", "RESTful", "AWS"
        ])
        self.assertEqual(result, "aws, dotnet, javascript, python, rest")

    def test_handles_none_and_empty(self):
        """Handles None values and empty strings gracefully."""
        result = normalize_hardskills([None, "", "python", None, ""])
        self.assertEqual(result, "python")


class TestNormalizeSoftskills(unittest.TestCase):
    """Tests for softskill normalization."""

    def test_empty_list(self):
        """Empty input returns empty string."""
        self.assertEqual(normalize_softskills([]), "")

    def test_basic_normalization(self):
        """Skills are lowercased and sorted."""
        result = normalize_softskills(["Leadership", "Teamwork"])
        self.assertEqual(result, "leadership, teamwork")

    def test_canonicalization_communication(self):
        """Communication variants are canonicalized."""
        result = normalize_softskills(["Communication", "written communication"])
        self.assertEqual(result, "communication skills")

    def test_canonicalization_teamwork(self):
        """Teamwork variants are canonicalized."""
        result = normalize_softskills(["team player", "team work", "team-oriented"])
        self.assertEqual(result, "teamwork")

    def test_canonicalization_collaboration(self):
        """Collaboration variants are canonicalized."""
        result = normalize_softskills(["cross-functional teams", "cross functional teams"])
        self.assertEqual(result, "collaboration")

    def test_mixed_softskills(self):
        """Mixed softskills are properly normalized."""
        result = normalize_softskills([
            "Team Player", "Communication", "Detail-oriented"
        ])
        self.assertEqual(result, "attention to detail, communication skills, teamwork")


if __name__ == "__main__":
    unittest.main()
