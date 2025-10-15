import importlib
import sys
from types import SimpleNamespace
from io import BytesIO

import pytest


@pytest.fixture()
def minio_module(monkeypatch):
    """Import the module under test with a mocked get_config to avoid real config access."""
    # Patch get_config before importing the module so CONFIG is created from our stub
    import nvidia_rag.utils.common as common

    dummy_config = SimpleNamespace(
        minio=SimpleNamespace(
            endpoint="dummy-endpoint:9000",
            access_key="dummy-access",
            secret_key="dummy-secret",
        )
    )
    monkeypatch.setattr(common, "get_config", lambda: dummy_config, raising=True)

    # Ensure any fake module injected by global test config is removed
    sys.modules.pop("nvidia_rag.utils.minio_operator", None)

    # Now import the real module under test
    import nvidia_rag.utils.minio_operator as minio_operator
    return minio_operator


class FakeMinioClient:
    def __init__(self, endpoint, access_key, secret_key, secure):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self._buckets = set()
        self.put_object_calls = []
        self.upload_snowball_calls = []
        self.get_object_response = None
        self.get_object_should_raise = False
        self.listed_objects = []
        self.removed_objects = []

    # Bucket ops
    def bucket_exists(self, bucket_name):
        return bucket_name in self._buckets

    def make_bucket(self, bucket_name):
        self._buckets.add(bucket_name)

    # Object ops
    def put_object(self, bucket, object_name, data, length, content_type=None):
        # Record for assertions
        self.put_object_calls.append(
            (bucket, object_name, data, length, content_type)
        )

    def upload_snowball_objects(self, bucket, snowball_objects):
        self.upload_snowball_calls.append((bucket, snowball_objects))

    def get_object(self, bucket, object_name):
        if self.get_object_should_raise:
            raise RuntimeError("get_object error")
        return self.get_object_response

    def list_objects(self, bucket, prefix="", recursive=False):
        class Obj:
            def __init__(self, name):
                self.object_name = name

        for name in self.listed_objects:
            yield Obj(name)

    def remove_object(self, bucket, object_name):
        self.removed_objects.append((bucket, object_name))


class FakeSnowballObject:
    def __init__(self, object_name, data, length):
        self.object_name = object_name
        self.data = data
        self.length = length


def test_constructor_creates_bucket_when_missing(monkeypatch, minio_module):
    # Patch Minio class used by the module
    client = FakeMinioClient(
        endpoint="ignored", access_key="", secret_key="", secure=False
    )

    def fake_minio_ctor(endpoint, access_key, secret_key, secure):
        # Assert ctor args
        assert endpoint == "dummy-endpoint:9000"
        assert access_key == "dummy-access"
        assert secret_key == "dummy-secret"
        assert secure is False
        return client

    monkeypatch.setattr(minio_module, "Minio", fake_minio_ctor, raising=True)

    # Bucket is missing initially
    operator = minio_module.MinioOperator(
        endpoint="dummy-endpoint:9000",
        access_key="dummy-access",
        secret_key="dummy-secret",
        default_bucket_name="default-bucket",
    )

    # Should have created the bucket
    assert "default-bucket" in client._buckets
    assert operator.default_bucket_name == "default-bucket"


def test_constructor_skips_bucket_creation_if_exists(monkeypatch, minio_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)
    client._buckets.add("existing-bucket")

    def fake_minio_ctor(endpoint, access_key, secret_key, secure):
        return client

    monkeypatch.setattr(minio_module, "Minio", fake_minio_ctor, raising=True)

    minio_module.MinioOperator(
        endpoint="e",
        access_key="a",
        secret_key="s",
        default_bucket_name="existing-bucket",
    )
    # No additional buckets should be created
    assert client._buckets == {"existing-bucket"}


def test_put_payload_uploads_json(monkeypatch, minio_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)

    monkeypatch.setattr(minio_module, "Minio", lambda *a, **k: client, raising=True)

    operator = minio_module.MinioOperator("e", "a", "s", default_bucket_name="b")
    payload = {"x": 1, "y": "z"}
    operator.put_payload(payload, object_name="obj.json")

    assert len(client.put_object_calls) == 1
    bucket, object_name, data, length, content_type = client.put_object_calls[0]
    assert bucket == "b"
    assert object_name == "obj.json"
    assert isinstance(data, BytesIO)
    assert content_type == "application/json"
    # Validate JSON roundtrip
    data.seek(0)
    body = data.read()
    assert length == len(body)
    assert body == b"{\"x\": 1, \"y\": \"z\"}"


def test_put_payloads_bulk_uses_snowball_objects(monkeypatch, minio_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)
    monkeypatch.setattr(minio_module, "Minio", lambda *a, **k: client, raising=True)
    # Patch SnowballObject used by module
    monkeypatch.setattr(minio_module, "SnowballObject", FakeSnowballObject, raising=True)

    operator = minio_module.MinioOperator("e", "a", "s", default_bucket_name="b")
    payloads = [{"a": 1}, {"b": 2}]
    names = ["o1.json", "o2.json"]
    operator.put_payloads_bulk(payloads, names)

    assert len(client.upload_snowball_calls) == 1
    bucket, snowballs = client.upload_snowball_calls[0]
    assert bucket == "b"
    assert [s.object_name for s in snowballs] == names
    # Ensure data content is correct JSON
    decoded = []
    for sb in snowballs:
        assert isinstance(sb.data, BytesIO)
        sb.data.seek(0)
        decoded.append(sb.data.read())
    assert decoded == [b"{\"a\": 1}", b"{\"b\": 2}"]


def test_get_payload_success(monkeypatch, minio_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)
    class Resp:
        def read(self):
            return b"{\"k\": \"v\"}"

    client.get_object_response = Resp()
    monkeypatch.setattr(minio_module, "Minio", lambda *a, **k: client, raising=True)

    operator = minio_module.MinioOperator("e", "a", "s", default_bucket_name="b")
    out = operator.get_payload("obj.json")
    assert out == {"k": "v"}


def test_get_payload_failure_returns_empty_dict(monkeypatch, minio_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)
    client.get_object_should_raise = True
    monkeypatch.setattr(minio_module, "Minio", lambda *a, **k: client, raising=True)

    operator = minio_module.MinioOperator("e", "a", "s", default_bucket_name="b")
    out = operator.get_payload("missing.json")
    assert out == {}


def test_list_and_delete_payloads(monkeypatch, minio_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)
    client.listed_objects = ["p1.json", "dir/p2.json"]
    monkeypatch.setattr(minio_module, "Minio", lambda *a, **k: client, raising=True)

    operator = minio_module.MinioOperator("e", "a", "s", default_bucket_name="b")

    listed = operator.list_payloads(prefix="dir/")
    assert listed == ["p1.json", "dir/p2.json"]

    operator.delete_payloads(["p1.json", "dir/p2.json"])
    assert client.removed_objects == [("b", "p1.json"), ("b", "dir/p2.json")]


def test_get_minio_operator_uses_config(monkeypatch, minio_module):
    # Provide our fake Minio so ctor is invoked as expected
    created_clients = []

    def fake_minio_ctor(endpoint, access_key, secret_key, secure):
        client = FakeMinioClient(endpoint, access_key, secret_key, secure)
        created_clients.append(client)
        return client

    monkeypatch.setattr(minio_module, "Minio", fake_minio_ctor, raising=True)

    op = minio_module.get_minio_operator(default_bucket_name="bucket-x")
    assert isinstance(op, minio_module.MinioOperator)
    assert len(created_clients) == 1
    c = created_clients[0]
    assert c.endpoint == "dummy-endpoint:9000"
    assert c.access_key == "dummy-access"
    assert c.secret_key == "dummy-secret"
    assert c.secure is False
    assert op.default_bucket_name == "bucket-x"


def test_unique_thumbnail_id_helpers(minio_module):
    p1 = minio_module.get_unique_thumbnail_id_collection_prefix("coll")
    assert p1 == "coll_::"

    p2 = minio_module.get_unique_thumbnail_id_file_name_prefix("coll", "file.pdf")
    # Note: current implementation adds an underscore before file name
    assert p2 == "coll_::_file.pdf_::"

    uid = minio_module.get_unique_thumbnail_id(
        collection_name="coll",
        file_name="file.pdf",
        page_number=3,
        location=[1.123456, 2.0, 3.987654, 4.5],
    )
    # Rounded to 4 decimals, and preserves current delimiter pattern
    assert uid == "coll_::_file.pdf_::_3_1.1235_2.0_3.9877_4.5"


