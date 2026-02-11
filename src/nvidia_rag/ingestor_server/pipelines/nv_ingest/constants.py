from nv_ingest_client.util.file_processing.extract import EXTENSION_TO_DOCUMENT_TYPE

SUPPORTED_FILE_TYPES = set(EXTENSION_TO_DOCUMENT_TYPE.keys()) - set({"svg"})
