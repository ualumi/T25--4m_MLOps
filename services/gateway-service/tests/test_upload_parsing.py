import json

from src.api import _parse_csv_clients, _parse_json_clients

VALID_FEATURES = [round(index * 0.1, 2) for index in range(1, 29)]


def test_parse_json_clients_accepts_array_payload() -> None:
    raw_content = json.dumps(
        [
            {"client_id": "c-1", "features": VALID_FEATURES},
            {"client_id": "c-2", "features": VALID_FEATURES},
        ]
    ).encode("utf-8")

    clients = _parse_json_clients(raw_content)

    assert len(clients) == 2
    assert clients[0]["client_id"] == "c-1"


def test_parse_json_clients_accepts_object_with_clients() -> None:
    raw_content = json.dumps(
        {"clients": [{"client_id": "c-1", "features": VALID_FEATURES}]}
    ).encode("utf-8")

    clients = _parse_json_clients(raw_content)

    assert len(clients) == 1
    assert clients[0]["client_id"] == "c-1"


def test_parse_json_clients_rejects_invalid_payload_shape() -> None:
    raw_content = json.dumps({"items": []}).encode("utf-8")

    try:
        _parse_json_clients(raw_content)
        assert False, "ValueError was expected"
    except ValueError as exc:
        assert "JSON file must be an array" in str(exc)


def test_parse_csv_clients_parses_rows() -> None:
    header = ["client_id", *[f"feature_{index}" for index in range(1, 29)]]
    row_one = ["c-1", *[str(value) for value in VALID_FEATURES]]
    row_two = ["c-2", *[str(value) for value in VALID_FEATURES]]
    raw_content = "\n".join(
        [
            ",".join(header),
            ",".join(row_one),
            ",".join(row_two),
        ]
    ).encode("utf-8")

    clients = _parse_csv_clients(raw_content)

    assert len(clients) == 2
    assert clients[0]["client_id"] == "c-1"
    assert clients[0]["features"] == VALID_FEATURES


def test_parse_csv_clients_rejects_missing_client_id_column() -> None:
    raw_content = "feature_1,feature_2\n0.1,0.2".encode("utf-8")

    try:
        _parse_csv_clients(raw_content)
        assert False, "ValueError was expected"
    except ValueError as exc:
        assert "client_id" in str(exc)


def test_parse_csv_clients_rejects_non_numeric_feature() -> None:
    header = ["client_id", "feature_1", "feature_2"]
    raw_content = "\n".join(
        [
            ",".join(header),
            "c-1,0.1,not-a-number",
        ]
    ).encode("utf-8")

    try:
        _parse_csv_clients(raw_content)
        assert False, "ValueError was expected"
    except ValueError as exc:
        assert "non-numeric feature values" in str(exc)
