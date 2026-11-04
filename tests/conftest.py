"""Offline defaults and small public-text inputs."""

import socket
from pathlib import Path

import pytest
import tiktoken

from rag_ai_act.ingest import CORPUS, Unit, chunk_units, clean_text, parse_units

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def pytest_sessionstart(session):
    # Warm the locally cached encoding before the per-test socket guard starts.
    tiktoken.get_encoding("cl100k_base")


def pytest_collection_modifyitems(config, items):
    text_dir = ROOT / "data" / "text"
    available = all((text_dir / Path(name).with_suffix(".txt")).is_file() for name in CORPUS)
    if not available:
        for item in items:
            if "corpus" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Corpus text is unavailable"))


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        return

    def blocked(*args, **kwargs):
        pytest.fail("Network access is forbidden in offline tests")

    for method in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, method, blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture
def snippet():
    def read(name):
        return (FIXTURES / name).read_text()

    return read


@pytest.fixture
def unit_factory():
    def make(source_id, text="Legal text", *, order=0, draft=False, header="Legal provision"):
        doc = source_id.split(":", 1)[0]
        return Unit(source_id, doc, None, header, text, draft, order)

    return make


@pytest.fixture
def mini_units(snippet):
    units = []
    for name, doc, part in (
        ("act_article6.txt", "AIA", None),
        ("act_article111.txt", "AIA", None),
        ("act_article113.txt", "AIA", None),
        ("guidelines_electricity.txt", "GLHR", "annex_III"),
    ):
        units.extend(parse_units(clean_text(snippet(name), doc), doc, part))
    return units


@pytest.fixture
def mini_chunks(mini_units):
    return chunk_units(mini_units)
