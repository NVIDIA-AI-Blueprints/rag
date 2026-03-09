#!/usr/bin/env python3
# main.py
"""Entry point for Kafka MinIO consumer."""

import logging

import config.settings as cfg
from config.constants import DEST_RAG
from services import ObjectStorage, DocumentIndexer
from handlers import DocumentHandler
from consumer import KafkaEventConsumer

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format=cfg.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def main():
    """Initialize and run the Kafka consumer."""
    logger.info("=" * 60)
    logger.info("Starting Kafka MinIO Consumer")
    logger.info("=" * 60)
    
    # Initialize services
    logger.info("Initializing services...")
    storage = ObjectStorage()
    indexer = DocumentIndexer(cfg.INGESTOR_SERVER_URL)

    # Initialize handlers
    logger.info("Initializing handlers...")
    handlers = {
        DEST_RAG: DocumentHandler(storage, indexer),
    }
    
    # Initialize consumer
    logger.info("Initializing Kafka consumer...")
    consumer = KafkaEventConsumer(handlers=handlers, storage=storage, history_file=cfg.HISTORY_FILE)
    
    # Run consumer loop
    logger.info("Starting consumer loop...")
    consumer.run()


if __name__ == '__main__':
    main()
