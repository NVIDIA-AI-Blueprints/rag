import importlib
import sys
from types import SimpleNamespace
from io import BytesIO
from pathlib import Path

import pytest


@pytest.fixture()
def object_store_module(monkeypatch):
    """Import the module under test with a mocked config."""
    # Ensure any fake module injected by global test config is removed
    sys.modules.pop("nvidia_rag.utils.object_store", None)

    # Now import the real module under test
    import nvidia_rag.utils.object_store as object_store

    return object_store


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


def test_constructor_creates_bucket_when_missing(monkeypatch, object_store_module):
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

    monkeypatch.setattr(object_store_module, "Minio", fake_minio_ctor, raising=True)

    # Bucket is missing initially
    operator = object_store_module.S3ObjectStoreOperator(
        endpoint="dummy-endpoint:9000",
        access_key="dummy-access",
        secret_key="dummy-secret",
        default_bucket_name="default-bucket",
    )

    # Should have created the bucket
    assert "default-bucket" in client._buckets
    assert operator.default_bucket_name == "default-bucket"


def test_constructor_skips_bucket_creation_if_exists(monkeypatch, object_store_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)
    client._buckets.add("existing-bucket")

    def fake_minio_ctor(endpoint, access_key, secret_key, secure):
        return client

    monkeypatch.setattr(object_store_module, "Minio", fake_minio_ctor, raising=True)

    object_store_module.S3ObjectStoreOperator(
        endpoint="e",
        access_key="a",
        secret_key="s",
        default_bucket_name="existing-bucket",
    )
    # No additional buckets should be created
    assert client._buckets == {"existing-bucket"}


def test_put_payload_uploads_json(monkeypatch, object_store_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)

    monkeypatch.setattr(
        object_store_module, "Minio", lambda *a, **k: client, raising=True
    )

    operator = object_store_module.S3ObjectStoreOperator(
        "e", "a", "s", default_bucket_name="b"
    )
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


def test_put_payloads_bulk_uses_snowball_objects(monkeypatch, object_store_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)
    monkeypatch.setattr(
        object_store_module, "Minio", lambda *a, **k: client, raising=True
    )
    # Patch SnowballObject used by module
    monkeypatch.setattr(
        object_store_module, "SnowballObject", FakeSnowballObject, raising=True
    )

    operator = object_store_module.S3ObjectStoreOperator(
        "e", "a", "s", default_bucket_name="b"
    )
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


def test_get_payload_success(monkeypatch, object_store_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)
    class Resp:
        def read(self):
            return b"{\"k\": \"v\"}"

    client.get_object_response = Resp()
    monkeypatch.setattr(
        object_store_module, "Minio", lambda *a, **k: client, raising=True
    )

    operator = object_store_module.S3ObjectStoreOperator(
        "e", "a", "s", default_bucket_name="b"
    )
    out = operator.get_payload("obj.json")
    assert out == {"k": "v"}


def test_get_payload_failure_returns_empty_dict(monkeypatch, object_store_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)
    client.get_object_should_raise = True
    monkeypatch.setattr(
        object_store_module, "Minio", lambda *a, **k: client, raising=True
    )

    operator = object_store_module.S3ObjectStoreOperator(
        "e", "a", "s", default_bucket_name="b"
    )
    out = operator.get_payload("missing.json")
    assert out == {}


def test_list_and_delete_payloads(monkeypatch, object_store_module):
    client = FakeMinioClient(endpoint="", access_key="", secret_key="", secure=False)
    client.listed_objects = ["p1.json", "dir/p2.json"]
    monkeypatch.setattr(
        object_store_module, "Minio", lambda *a, **k: client, raising=True
    )

    operator = object_store_module.S3ObjectStoreOperator(
        "e", "a", "s", default_bucket_name="b"
    )

    listed = operator.list_payloads(prefix="dir/")
    assert listed == ["p1.json", "dir/p2.json"]

    operator.delete_payloads(["p1.json", "dir/p2.json"])
    assert client.removed_objects == [("b", "p1.json"), ("b", "dir/p2.json")]


def test_get_object_store_operator_uses_config(monkeypatch, object_store_module):
    # Provide our fake Minio so ctor is invoked as expected
    created_clients = []

    def fake_minio_ctor(endpoint, access_key, secret_key, secure):
        client = FakeMinioClient(endpoint, access_key, secret_key, secure)
        created_clients.append(client)
        return client

    monkeypatch.setattr(object_store_module, "Minio", fake_minio_ctor, raising=True)

    # Create a mock config with the expected object-store settings
    from types import SimpleNamespace
    from pydantic import SecretStr
    
    mock_config = SimpleNamespace(
        object_store=SimpleNamespace(
            backend="s3",
            endpoint="dummy-endpoint:9000",
            access_key=SecretStr("dummy-access"),
            secret_key=SecretStr("dummy-secret"),
            secure=False,
        )
    )

    op = object_store_module.get_object_store_operator(
        default_bucket_name="bucket-x", config=mock_config
    )
    assert isinstance(op, object_store_module.S3ObjectStoreOperator)
    assert len(created_clients) == 1
    c = created_clients[0]
    assert c.endpoint == "dummy-endpoint:9000"


def test_filesystem_operator_roundtrip(tmp_path, object_store_module):
    operator = object_store_module.FilesystemObjectStoreOperator(
        root_path=tmp_path / "object-store",
        default_bucket_name="bucket-a",
    )

    operator.put_payload({"hello": "world"}, "dir/test.json")

    assert operator.get_payload("dir/test.json") == {"hello": "world"}
    assert operator.get_object("dir/test.json") == b'{"hello": "world"}'
    assert operator.list_payloads("dir/") == ["dir/test.json"]

    stored_uri = (tmp_path / "object-store" / "bucket-a" / "dir" / "test.json").as_uri()
    assert operator.get_object_from_uri(stored_uri) == b'{"hello": "world"}'

    operator.delete_payloads(["dir/test.json"])
    assert operator.list_payloads() == []


def test_get_object_store_operator_returns_filesystem_backend(tmp_path, object_store_module):
    from pydantic import SecretStr

    mock_config = SimpleNamespace(
        object_store=SimpleNamespace(
            backend="filesystem",
            storage_root=Path(tmp_path / "filesystem-store").resolve(),
            endpoint="unused:9010",
            access_key=SecretStr("dummy-access"),
            secret_key=SecretStr("dummy-secret"),
            secure=False,
        )
    )

    op = object_store_module.get_object_store_operator(
        default_bucket_name="bucket-x",
        config=mock_config,
    )

    assert isinstance(op, object_store_module.FilesystemObjectStoreOperator)
    assert op.root_path == Path(tmp_path / "filesystem-store").resolve()
    assert op.default_bucket_name == "bucket-x"


def test_unique_thumbnail_id_helpers(object_store_module):
    p1 = object_store_module.get_unique_thumbnail_id_collection_prefix("coll")
    assert p1 == "coll_::"

    p2 = object_store_module.get_unique_thumbnail_id_file_name_prefix("coll", "file.pdf")
    # Note: current implementation adds an underscore before file name
    assert p2 == "coll_::_file.pdf_::"

    uid = object_store_module.get_unique_thumbnail_id(
        collection_name="coll",
        file_name="file.pdf",
        page_number=3,
        location=[1.123456, 2.0, 3.987654, 4.5],
    )
    # Rounded to 4 decimals, and preserves current delimiter pattern
    assert uid == "coll_::_file.pdf_::_3_1.1235_2.0_3.9877_4.5"


def test_extract_location_from_metadata_from_content_metadata(object_store_module):
    """Test extract_location_from_metadata when location is in content_metadata"""
    metadata = {
        "content_metadata": {
            "type": "image",
            "location": [10.0, 20.0, 30.0, 40.0]
        }
    }
    location = object_store_module.extract_location_from_metadata("image", metadata)
    assert location == [10.0, 20.0, 30.0, 40.0]


def test_extract_location_from_metadata_from_image_metadata(object_store_module):
    """Test extract_location_from_metadata from image_metadata"""
    metadata = {
        "image_metadata": {
            "image_location": [5.0, 10.0, 15.0, 20.0]
        }
    }
    location = object_store_module.extract_location_from_metadata("image", metadata)
    assert location == [5.0, 10.0, 15.0, 20.0]


def test_extract_location_from_metadata_from_table_metadata(object_store_module):
    """Test extract_location_from_metadata from table_metadata"""
    metadata = {
        "table_metadata": {
            "table_location": [1.0, 2.0, 3.0, 4.0]
        }
    }
    location = object_store_module.extract_location_from_metadata("structured", metadata)
    assert location == [1.0, 2.0, 3.0, 4.0]


def test_extract_location_from_metadata_from_chart_metadata(object_store_module):
    """Test extract_location_from_metadata from chart_metadata"""
    metadata = {
        "chart_metadata": {
            "chart_location": [11.0, 22.0, 33.0, 44.0]
        }
    }
    location = object_store_module.extract_location_from_metadata("structured", metadata)
    assert location == [11.0, 22.0, 33.0, 44.0]


def test_extract_location_from_metadata_no_location(object_store_module):
    """Test extract_location_from_metadata when no location is found"""
    metadata = {"some_other_field": "value"}
    location = object_store_module.extract_location_from_metadata("image", metadata)
    assert location == []


def test_get_unique_thumbnail_id_from_result_with_location(object_store_module):
    """Test get_unique_thumbnail_id_from_result with location provided"""
    result = object_store_module.get_unique_thumbnail_id_from_result(
        collection_name="test_coll",
        file_name="doc.pdf",
        page_number=1,
        location=[10.0, 20.0, 30.0, 40.0],
        metadata=None
    )
    assert result == "test_coll_::_doc.pdf_::_1_10.0_20.0_30.0_40.0"


def test_get_unique_thumbnail_id_from_result_with_metadata_fallback(object_store_module):
    """Test get_unique_thumbnail_id_from_result with metadata fallback"""
    metadata = {
        "content_metadata": {
            "type": "image",
            "location": [5.0, 10.0, 15.0, 20.0]
        }
    }
    result = object_store_module.get_unique_thumbnail_id_from_result(
        collection_name="test_coll",
        file_name="doc.pdf",
        page_number=2,
        location=None,
        metadata=metadata
    )
    assert result == "test_coll_::_doc.pdf_::_2_5.0_10.0_15.0_20.0"


def test_get_unique_thumbnail_id_from_result_no_location_returns_none(object_store_module):
    """Test get_unique_thumbnail_id_from_result returns None when no location found"""
    result = object_store_module.get_unique_thumbnail_id_from_result(
        collection_name="test_coll",
        file_name="doc.pdf",
        page_number=1,
        location=None,
        metadata={}
    )
    assert result is None


def test_get_unique_thumbnail_id_from_result_handles_exceptions(object_store_module):
    """Test get_unique_thumbnail_id_from_result handles exceptions gracefully"""
    # Provide invalid location that would cause an error in get_unique_thumbnail_id
    result = object_store_module.get_unique_thumbnail_id_from_result(
        collection_name="test_coll",
        file_name="doc.pdf",
        page_number=1,
        location="invalid_location",  # Not a list
        metadata=None
    )
    # Should return None on exception
    assert result is None
