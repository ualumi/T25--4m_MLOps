from src.infrastructure.postgres.session_store import PostgresSessionStore


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.fetchone_result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, query, params=None) -> None:
        self.connection.queries.append((" ".join(query.split()), params))
        if "WHERE user_id = %s AND token = %s" in query:
            user_id, token = params
            if self.connection.tokens.get(user_id) == token:
                self.fetchone_result = (1,)
            else:
                self.fetchone_result = None
        elif "INSERT INTO user_sessions" in query:
            user_id, token = params
            self.connection.tokens[user_id] = token

    def fetchone(self):
        return self.fetchone_result


class FakeConnection:
    def __init__(self):
        self.tokens = {}
        self.queries = []
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def cursor(self):
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


def test_postgres_session_store_create_and_validate(monkeypatch) -> None:
    connection = FakeConnection()
    database_url = "postgresql://postgres:postgres@localhost:5432/gateway_db"

    def fake_connect(url: str):
        assert url == database_url
        return connection

    monkeypatch.setattr(
        "src.infrastructure.postgres.session_store.psycopg.connect",
        fake_connect,
    )

    store = PostgresSessionStore(database_url)
    session = store.create("u-1")

    assert session.user_id == "u-1"
    assert store.is_valid("u-1", session.token) is True
    assert store.is_valid("u-1", "wrong-token") is False
    assert connection.commits == 2
