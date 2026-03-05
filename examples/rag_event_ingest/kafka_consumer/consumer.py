# consumer.py
"""Kafka consumer for MinIO S3 events."""

import json
import logging
from datetime import datetime
from typing import Dict, Optional
from kafka import KafkaConsumer

import config.settings as cfg
from config.constants import DEST_RAG, DEST_SKIP, STATUS_FAILED, KEY_DESTINATION, KEY_FILE_TYPE, KEY_REASON
from router import FileRouter
from models.events import S3Event, HandlerResult, IngestionRecord
from handlers.base import BaseHandler
from services.storage import ObjectStorage

logger = logging.getLogger(__name__)


class KafkaEventConsumer:
    """Kafka consumer that routes MinIO events to handlers."""
    
    def __init__(
        self,
        handlers: Dict[str, BaseHandler],
        storage: ObjectStorage,
        history_file: str = '/tmp/ingestion_history.jsonl'
    ):
        """Initialize Kafka consumer."""
        self.handlers = handlers
        self.storage = storage
        self.history_file = history_file
        self.router = FileRouter()
        
        logger.info(f"Connecting to Kafka: {cfg.KAFKA_BOOTSTRAP_SERVERS}")
        logger.info(f"Consumer group: {cfg.KAFKA_CONSUMER_GROUP}")
        
        self.kafka_consumer = KafkaConsumer(
            cfg.KAFKA_TOPIC,
            bootstrap_servers=cfg.KAFKA_BOOTSTRAP_SERVERS.split(','),
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=cfg.KAFKA_CONSUMER_GROUP,
            auto_offset_reset=cfg.KAFKA_AUTO_OFFSET_RESET,
            enable_auto_commit=True,
            max_poll_records=cfg.KAFKA_MAX_POLL_RECORDS,
            max_poll_interval_ms=cfg.KAFKA_MAX_POLL_INTERVAL_MS,
            session_timeout_ms=cfg.KAFKA_SESSION_TIMEOUT_MS,
            heartbeat_interval_ms=cfg.KAFKA_HEARTBEAT_INTERVAL_MS
        )
        
        logger.info("Kafka consumer initialized")
        logger.info(f"Registered handlers: {list(self.handlers.keys())}")
    
    def process_event(self, raw_event: dict) -> Optional[HandlerResult]:
        """Process a single MinIO S3 event."""
        start_time = datetime.now()
        event: Optional[S3Event] = None
        result: Optional[HandlerResult] = None
        
        try:
            logger.info(f"Received event: {json.dumps(raw_event, indent=2)}")
            
            event = S3Event.from_kafka_message(
                raw_event,
                collection_resolver=self.storage.get_collection_for_bucket
            )
            
            if not event:
                logger.warning("Invalid event format, skipping")
                return None
            
            logger.info(f"Processing: {event.bucket}/{event.key} ({event.size} bytes)")
            
            if event.event_type == 'delete':
                result = self._handle_delete(event)
            else:
                result = self._handle_create(event)
            
            return result
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Invalid event data: {e}")
            result = HandlerResult.failed_result(str(e))
            return result
            
        except (IOError, OSError) as e:
            logger.error(f"Storage error: {e}")
            result = HandlerResult.failed_result(str(e))
            return result
            
        finally:
            if event:
                self._save_record(event, result, start_time)
    
    def _handle_delete(self, event: S3Event) -> HandlerResult:
        """Handle S3 delete event."""
        logger.info(f"🗑️  DELETE event for {event.key}")
        
        doc_handler = self.handlers.get(DEST_RAG)
        if doc_handler and hasattr(doc_handler, 'indexer'):
            success = doc_handler.indexer.delete(event.key, event.collection)
            if success:
                logger.info(f"✓ Deleted {event.key} from Milvus")
                return HandlerResult(success=True, status='DELETED')
        
        return HandlerResult.failed_result("Delete failed")
    
    def _handle_create(self, event: S3Event) -> HandlerResult:
        """Handle S3 create event."""
        route_info = self.router.route(event.key)
        destination = route_info[KEY_DESTINATION]
        
        logger.info(f"📁 {route_info[KEY_FILE_TYPE]} → {destination}")
        
        if destination == DEST_SKIP:
            reason = route_info.get(KEY_REASON, 'Skipped by router')
            logger.info(f"⏭️  Skipping: {reason}")
            return HandlerResult.skipped_result(reason)
        
        handler = self.handlers.get(destination)
        if not handler:
            handler = self.handlers.get(DEST_RAG)
        
        if not handler:
            return HandlerResult.failed_result(f"No handler for {destination}")
        
        return handler.handle(event)
    
    def _save_record(self, event: S3Event, result: Optional[HandlerResult], start_time: datetime):
        """Save ingestion record to history file."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        record = IngestionRecord(
            file_name=event.key,
            bucket=event.bucket,
            collection=event.collection,
            status=result.status if result else STATUS_FAILED,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            error_message=result.error_message if result else None,
            task_id=result.task_id if result else None
        )
        
        try:
            with open(self.history_file, 'a') as f:
                f.write(json.dumps(record.to_dict()) + '\n')
        except (IOError, OSError) as e:
            logger.error(f"Failed to save history: {e}")
        
        status_emoji = '✓' if record.status in ['SUCCESS', 'DELETED', 'SKIPPED'] else '✗'
        logger.info(
            f"{status_emoji} SUMMARY: {event.key} | "
            f"Collection: {event.collection} | "
            f"Duration: {duration:.2f}s | "
            f"Status: {record.status}"
        )
    
    def run(self):
        """Main consumer loop."""
        logger.info("Starting Kafka consumer loop...")
        logger.info(f"Subscribed topics: {self.kafka_consumer.subscription()}")
        logger.info("Waiting for messages...")
        
        try:
            message_count = 0
            for message in self._poll_messages():
                message_count += 1
                logger.info(
                    f"[{message_count}] Message from "
                    f"partition {message.partition}, offset {message.offset}"
                )
                self.process_event(message.value)
                            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.kafka_consumer.close()
            logger.info("Consumer closed")
    
    def _poll_messages(self):
        """Generator that yields messages from Kafka."""
        while True:
            msg_pack = self.kafka_consumer.poll(timeout_ms=5000, max_records=1)
            
            if not msg_pack:
                logger.debug("No messages, continuing...")
                continue
            
            for messages in msg_pack.values():
                yield from messages
