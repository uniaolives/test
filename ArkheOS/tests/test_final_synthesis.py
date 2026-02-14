import pytest
from arkhe.unification import UniqueVocabulary

def test_unique_vocabulary():
    assert UniqueVocabulary.translate("neuron") == "Direction 1: Coherence (C)"
    assert UniqueVocabulary.translate("melanocyte") == "Direction 2: Fluctuation (F)"
    assert UniqueVocabulary.translate("synapse") == "Inner Product ⟨i|j⟩ (Syzygy)"

    report = UniqueVocabulary.get_hermeneutic_report()
    assert report['State'] == "Γ_∞+57" # Updated for Γ_∞+57

def test_multidisciplinary_vocabulary():
    assert "SPRTN" in UniqueVocabulary.translate("sprtn_enzyme")
    assert "cGAS-STING" in UniqueVocabulary.translate("cgas_sting")
    assert "Klein" in UniqueVocabulary.translate("gluon_amplitude")
    assert "ERP" in UniqueVocabulary.translate("erp")

def test_hermeneutic_report():
    report = UniqueVocabulary.get_hermeneutic_report()
    assert report["Vocabulary"] == "Unified (Triune & Lysosomal)"
