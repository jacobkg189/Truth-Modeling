import json

import pytest

from factchecker.reliability import DEFAULT_PRIOR, SiteStance, SourceReliability, site_of


@pytest.mark.parametrize(
    ("url", "site"),
    [
        ("https://en.wikipedia.org/wiki/Marie_Curie", "wikipedia.org"),
        ("https://pmc.ncbi.nlm.nih.gov/articles/PMC1", "nih.gov"),
        ("https://www.bbc.co.uk/news/x", "bbc.co.uk"),
        ("http://foo.blogspot.com/2020/post", "blogspot.com"),
        ("https://www.ox.ac.uk/research", "ox.ac.uk"),
        ("https://REUTERS.com./world", "reuters.com"),
        ("/Users/me/docs/policy.md", "local"),
        (None, "local"),
    ],
)
def test_site_of(url, site):
    assert site_of(url, fallback="local") == site


def test_priors_by_site_suffix_and_default():
    r = SourceReliability()
    assert r.weight("wikipedia.org") == pytest.approx(0.85)
    assert r.weight("nih.gov") == pytest.approx(0.9)
    assert r.weight("ox.ac.uk") == pytest.approx(0.85)
    assert r.weight("slideshare.net") == pytest.approx(0.3)
    assert r.weight("some-random-site.com") == pytest.approx(DEFAULT_PRIOR)
    assert r.weight("notagov.com") == pytest.approx(DEFAULT_PRIOR)


def test_weight_moves_gradually_and_stays_bounded():
    r = SourceReliability(prior_strength=20)
    site = "new-site.com"
    r.record(site, agreed=False)
    once = r.weight(site)
    assert DEFAULT_PRIOR - 0.05 < once < DEFAULT_PRIOR  # one claim only nudges it
    for _ in range(1000):
        r.record(site, agreed=False)
    assert r.weight(site) == pytest.approx(r.floor)
    for _ in range(2000):
        r.record("good.com", agreed=True)
    assert r.weight("good.com") == pytest.approx(r.ceiling)
    r.record("ignored.com", agreed=True, strength=0)
    assert "ignored.com" not in r.records


def test_learn_from_label_scores_only_sites_that_took_a_side():
    r = SourceReliability()
    stances = [
        SiteStance("right.com", supports=0.9, contradicts=0.0),
        SiteStance("wrong.com", supports=0.0, contradicts=0.8),
        SiteStance("neutral.com", supports=0.1, contradicts=0.1),
    ]
    assert r.learn_from_label(stances, label_is_true=True, min_side=0.6) == 2
    assert r.records["right.com"].agreements == pytest.approx(0.9)
    assert r.records["wrong.com"].disagreements == pytest.approx(0.8)
    assert "neutral.com" not in r.records


def test_save_load_roundtrip_and_corrupt_file(tmp_path):
    path = tmp_path / "w.json"
    r = SourceReliability(path)
    r.record("a.com", agreed=True, strength=2)
    r.record("a.com", agreed=False)
    r.save()
    loaded = SourceReliability(path)
    assert loaded.weight("a.com") == pytest.approx(r.weight("a.com"))
    assert json.loads(path.read_text())["sites"]["a.com"] == {"agreements": 2.0, "disagreements": 1.0}
    assert loaded.table()[0][0] == "a.com"

    path.write_text("{not json")
    assert SourceReliability(path).records == {}  # a corrupt file never blocks checking


def test_save_is_noop_without_changes(tmp_path):
    path = tmp_path / "w.json"
    SourceReliability(path).save()
    assert not path.exists()
