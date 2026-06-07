from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

PROMPT = """
Give me unit test for this code:
Directory structure:
└── starsin-face-kyc-consumer/
    ├── requirements.txt
    ├── .env.example
    ├── app/
    │   ├── config.py
    │   ├── consumer.py
    │   ├── health.py
    │   ├── kafka_helpers.py
    │   ├── logging_config.py
    │   ├── main.py
    │   ├── metrics.py
    │   └── schema.py
    └── docker/
        ├── docker-compose.yml
        └── Dockerfile

================================================
FILE: requirements.txt
================================================
confluent-kafka
httpx
msgspec
python-dotenv
pyyaml
python-json-logger
prometheus-client



================================================
FILE: .env.example
================================================
# ------------------------- Kafka Topic names -------------------------
# MEDIA_KYC_CREATED: Kafka topic for new KYC submissions
MEDIA_KYC_CREATED=mcs.kyc_submission.created
# MEDIA_KYC_COMPLETED: Kafka topic for completed KYC
MEDIA_KYC_COMPLETED=ai.kyc_submission.verified
# MEDIA_KYC_FAILED: Kafka topic for failed KYC
MEDIA_KYC_FAILED=ai.kyc_submission.failed
# MEDIA_KYC_EMBEDDINGS: Kafka topic for KYC embeddings
MEDIA_KYC_EMBEDDINGS=media.kyc.embeddings


# --------------------- Kafka connection settings ---------------------
# KAFKA_AUTO_OFFSET_RESET: Where to start if no offset default:earliest
KAFKA_AUTO_OFFSET_RESET=earliest
# KAFKA_ENABLE_AUTO_COMMIT: Enable/disable auto commit
KAFKA_ENABLE_AUTO_COMMIT=false
# KAFKA_SASL_MECHANISM: SASL mechanism
KAFKA_SASL_MECHANISM=SCRAM-SHA-256
# KAFKA_SECURITY_PROTOCOL: SASL_SSL
KAFKA_SECURITY_PROTOCOL=PLAINTEXT
# KAFKA_USERNAME: Kafka username
KAFKA_USERNAME=your_username
# KAFKA_PASSWORD: Kafka password
KAFKA_PASSWORD=your_password
# KAFKA_MAX_POLL_INTERVAL_MS: Max poll interval (ms)
KAFKA_MAX_POLL_INTERVAL_MS=600000
# KAFKA_SESSION_TIMEOUT_MS: Session timeout (ms)
KAFKA_SESSION_TIMEOUT_MS=45000
# MAX_CONCURRENCY: Max in-flight Kafka messages
MAX_CONCURRENCY=5


# ---------------------- HTTP Connection settings ---------------------
# HTTP_TIMEOUT: Timeout for HTTP calls (seconds)
HTTP_TIMEOUT=120
# HTTP_MAX_CONNECTIONS: Max HTTP connections
HTTP_MAX_CONNECTIONS=20
# HTTP_MAX_KEEPALIVE: Max keep-alive connections
HTTP_MAX_KEEPALIVE=10
# Face Service Endpoint
FACE_SERVICE_URL=http://face-service:8001

# HEALTH_PORT: Health server port
HEALTH_PORT=8001


================================================
FILE: app/config.py
================================================
import os
import socket
from dotenv import load_dotenv

load_dotenv()


class Settings:
    CONSUMER_ID: str = os.getenv("CONSUMER_ID", socket.gethostname())

    KAFKA = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP", "kafka:29092"),
        "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
    }
    if os.getenv("KAFKA_USERNAME"):
        KAFKA.update({
            "sasl.mechanisms": os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256"),
            "sasl.username": os.getenv("KAFKA_USERNAME"),
            "sasl.password": os.getenv("KAFKA_PASSWORD"),
        })
    KAFKA_CONSUMER = {
        **KAFKA,
        "group.id": os.getenv("KAFKA_GROUP_ID", "ai-kyc-processor"),
        "auto.offset.reset": os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest"),
        "enable.auto.commit": False,
        "client.id": f"c-{os.getenv('CONSUMER_ID', socket.gethostname())}",
        "max.poll.interval.ms": int(os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", "600000")),
        "session.timeout.ms": int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "45000")),
    }
    KAFKA_PRODUCER = {
        **KAFKA,
        "client.id": f"p-{os.getenv('CONSUMER_ID', socket.gethostname())}",
        "acks": "all",  # Wait for all in-sync replicas to acknowledge
        "retries": 10,  # Retry failed sends up to 10 times
        "max.in.flight.requests.per.connection": 5,  # Allow 5 concurrent requests
        "request.timeout.ms": 30000,  # 30 second timeout for broker requests
    }

    FACE_SERVICE_URL: str = os.getenv("FACE_SERVICE_URL", "http://face-service:8001")
    HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "120"))
    HTTP_MAX_CONNECTIONS: int = int(os.getenv("HTTP_MAX_CONNECTIONS", "20"))
    HTTP_MAX_KEEPALIVE: int = int(os.getenv("HTTP_MAX_KEEPALIVE", "10"))

    MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", "5"))
    HEALTH_PORT: int = int(os.getenv("HEALTH_PORT", "8081"))

    MEDIA_KYC_CREATED: str = os.getenv("MEDIA_KYC_CREATED", "mcs.kyc_submission.created")
    MEDIA_KYC_COMPLETED: str = os.getenv("MEDIA_KYC_COMPLETED", "ai.kyc_submission.verified")
    MEDIA_KYC_FAILED: str = os.getenv("MEDIA_KYC_FAILED", "ai.kyc_submission.failed")
    MEDIA_KYC_EMBEDDINGS: str = os.getenv("MEDIA_KYC_EMBEDDINGS", "media.kyc.embeddings")


settings = Settings()



================================================
FILE: app/consumer.py
================================================
from __future__ import annotations

import json
import time
import httpx
import msgspec
import logging
from app.config import settings
from msgspec.json import encode
from app.kafka_helpers import ReliableProducer
from app.metrics import (
    messages_consumed_total,
    message_processing_duration_seconds,
    http_client_request_duration_seconds,
    producer_publish_total,
)
from app.schema import (
    KYCEvent,
    KYCCompletedData,
    FailurePayload,
    ErrorInfo,
)


logger = logging.getLogger(__name__)
_SVC = "kyc-consumer"

async def process_event(msg, producer: ReliableProducer, http_client: httpx.AsyncClient) -> None:
    event_id = video_id = user_id = key = "UNKNOWN"
    topic = settings.MEDIA_KYC_CREATED
    start_time = time.perf_counter()
    try:
        parsed = json.loads(msg.value().decode('utf-8')) 
        if isinstance(parsed, str): 
            parsed = json.loads(parsed)
        payload = msgspec.convert(parsed, type=KYCEvent)
        event_id = payload.event_id
        data = payload.data
        video_id = data.video_id
        user_id = data.user_id
        bucket = data.kyc_video_storage.bucket
        key = data.kyc_video_storage.s3_key
        logger.info("Processing KYC event", extra={"video_id": video_id, "event_id": event_id, "user_id": user_id})
        # ── Call Face Service ────────────────────────────────────────
        t0 = time.perf_counter()
        resp = await http_client.post(
            f"{settings.FACE_SERVICE_URL}/extract-embeddings",
            json={
                "bucket": bucket,
                "s3_key": key,
                "num_samples": 10,
                "uniform": True,
            },
            timeout=settings.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        http_client_request_duration_seconds.labels(
            service=_SVC, target_service="face-service",
            endpoint="/extract-embeddings", status=resp.status_code,
        ).observe(time.perf_counter() - t0)

        embeddings = resp.json()["embeddings"]

        # ── Publish embeddings (JDBC Sink format) ────────────────────
        kafka_payload = {
            "schema": {
                "type": "struct",
                "fields": [
                        {"type": "string", "field": "user_id", "optional": False},
                        {"type": "string", "field": "embeddings", "optional": False},
                    ],
                "optional": False,
                },
                "payload": {
                    "user_id": user_id,
                    "embeddings": json.dumps(embeddings),
                },
            }
        await producer.publish_and_confirm(settings.MEDIA_KYC_EMBEDDINGS, json.dumps(kafka_payload).encode("utf-8"))
        producer_publish_total.labels(service=_SVC, topic=settings.MEDIA_KYC_EMBEDDINGS, status="success").inc()
        logger.info("Embeddings published", extra={"video_id": video_id, "event_id": event_id, "user_id": user_id})

        # ── Publish completion (atomic -- only after embeddings confirmed) ──
        processing_time = int((time.perf_counter() - start_time) * 1000)
        completed = KYCCompletedData(
            event_id=event_id,
            video_id=video_id,
            user_id=user_id,
            kyc_subtype="verification",
            status="SUCCESS",
            processing_time_ms=processing_time,
        )
        await producer.publish(settings.MEDIA_KYC_COMPLETED, encode(completed))
        producer_publish_total.labels(service=_SVC, topic=settings.MEDIA_KYC_COMPLETED, status="success").inc()

        elapsed = time.perf_counter() - start_time
        messages_consumed_total.labels(service=_SVC, topic=topic, status="success").inc()
        message_processing_duration_seconds.labels(service=_SVC, topic=topic).observe(elapsed)
        logger.info("KYC COMPLETED", extra={"video_id": video_id, "event_id": event_id, "user_id": user_id, "processing_time_ms": processing_time})

    except msgspec.ValidationError as exc:
        messages_consumed_total.labels(service=_SVC, topic=topic, status="failure").inc()
        logger.error("Schema validation failed", extra={"error": str(exc)})
        failure = FailurePayload(
            event_id=event_id,
            video_id=video_id,
            user_id=user_id,
            s3_key=key,
            status="FAILED",
            error=ErrorInfo(
                code="ERR_SCHEMA_VALIDATION_FAILED",
                message=str(exc),
                is_retryable=False,
            ),
        )
        await producer.publish(settings.MEDIA_KYC_FAILED, encode(failure))

    except httpx.HTTPStatusError as exc:
        messages_consumed_total.labels(service=_SVC, topic=topic, status="failure").inc()
        code = "ERR_FACE_SERVICE_ERROR"
        retryable = exc.response.status_code >= 500
        logger.error("Face service HTTP error", extra={"video_id": video_id, "status_code": exc.response.status_code})
        failure = FailurePayload(
            event_id=event_id,
            video_id=video_id,
            user_id=user_id,
            s3_key=key,
            status="FAILED",
            error=ErrorInfo(code=code, message=str(exc), is_retryable=retryable),
        )
        await producer.publish(settings.MEDIA_KYC_FAILED, encode(failure))

    except (httpx.RequestError, httpx.TimeoutException) as exc:
        messages_consumed_total.labels(service=_SVC, topic=topic, status="failure").inc()
        logger.error("Face service unreachable", extra={"video_id": video_id, "error": str(exc)})
        failure = FailurePayload(
            event_id=event_id,
            video_id=video_id,
            user_id=user_id,
            s3_key=key,
            status="FAILED",
            error=ErrorInfo(
                code="ERR_FACE_SERVICE_UNAVAILABLE",
                message=str(exc),
                is_retryable=True,
            ),
        )
        await producer.publish(settings.MEDIA_KYC_FAILED, encode(failure))

    except Exception as exc:
        messages_consumed_total.labels(service=_SVC, topic=topic, status="failure").inc()
        logger.exception("Unexpected error %s", exc, extra={"video_id": video_id, "event_id": event_id, "user_id": user_id})
        failure = FailurePayload(
            event_id=event_id,
            video_id=video_id,
            user_id=user_id,
            s3_key=key,
            status="FAILED",
            error=ErrorInfo(
                code="ERR_UNEXPECTED",
                message=str(exc),
                is_retryable=False,
            ),
        )
        await producer.publish(settings.MEDIA_KYC_FAILED, encode(failure))



================================================
FILE: app/health.py
================================================
from __future__ import annotations

import asyncio
import logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

_RESPONSE_OK = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: text/plain\r\n"
    b"Content-Length: 2\r\n"
    b"Connection: close\r\n"
    b"\r\n"
    b"OK"
)


def _build_metrics_response() -> bytes:
    body = generate_latest()
    header = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: " + CONTENT_TYPE_LATEST.encode() + b"\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n"
        b"\r\n"
    )
    return header + body


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        request_line = await reader.readline()
        # Drain remaining headers
        await reader.read(4096)

        if b"/metrics" in request_line:
            try:
                writer.write(_build_metrics_response())
            except Exception as e:
                logger.exception("Failed to build metrics response: %s", e)
                writer.write(_RESPONSE_OK)
        else:
            writer.write(_RESPONSE_OK)
        await writer.drain()
    except Exception as exc:
        logger.exception("Exception in health server handler: %s", exc)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception as close_exc:
            logger.warning("Exception while closing writer: %s", close_exc)


async def start_health_server(port: int = 8080) -> asyncio.Server:
    server = await asyncio.start_server(_handle, "0.0.0.0", port)
    logger.info("Health/metrics server listening on :%d", port)
    return server



================================================
FILE: app/kafka_helpers.py
================================================
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable, Any

from confluent_kafka.aio import AIOProducer, AIOConsumer

logger = logging.getLogger(__name__)


class ReliableProducer:
    def __init__(self, kafka_config: dict, max_retries: int = 3):
        self._config = kafka_config
        self._producer: AIOProducer | None = None
        self._max_retries = max_retries

    async def _ensure(self) -> AIOProducer:
        if self._producer is None:
            self._producer = AIOProducer(self._config)
        return self._producer

    async def publish(self, topic: str, payload: bytes, on_delivery: Callable | None = None) -> None:
        producer = await self._ensure()
        for attempt in range(1, self._max_retries + 1):
            try:
                await producer.produce(topic, payload, callback=on_delivery)
                return
            except BufferError:
                logger.warning("Producer buffer full (attempt %d/%d), flushing…", attempt, self._max_retries)
                await producer.flush()
        await producer.produce(topic, payload, callback=on_delivery)

    async def publish_and_confirm(self, topic: str, payload: bytes) -> Any:

        producer = await self._ensure()
        await self.publish(topic, payload, on_delivery=None)
        
        try:
            logger.debug("Flushing producer to confirm delivery", extra={"topic": topic})
            await producer.flush()  # 30 second timeout
            logger.debug("Producer flushed successfully", extra={"topic": topic})
            return None
        except Exception as e:
            logger.error("Producer flush failed", extra={"topic": topic, "error": str(e)})
            raise TimeoutError(f"Message delivery to {topic} timed out: {e}")

    async def flush(self) -> None:
        if self._producer:
            await self._producer.flush()


class ConsumerLoop:
    def __init__(self, kafka_config: dict, topics: list[str], max_concurrency: int = 5):
        self._config = kafka_config
        self._topics = topics
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._partition_locks: dict[int, asyncio.Lock] = {}
        self._consumer: AIOConsumer | None = None
        self._running = False

    async def start(self, handler: Callable[[Any], Awaitable[None]]) -> None:
        self._consumer = AIOConsumer(self._config)
        await self._consumer.subscribe(self._topics)
        self._running = True
        logger.info("Consumer subscribed to %s", self._topics)

        tasks: set[asyncio.Task] = set()
        try:
            while self._running:
                msg = await self._consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.error("Consumer poll error: %s", msg.error())
                    continue
                await self._semaphore.acquire()
                task = asyncio.create_task(self._process(handler, msg))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except asyncio.CancelledError:
            logger.info("Consumer shutting down, draining %d in-flight tasks…", len(tasks))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if self._consumer:
                await self._consumer.close()
            logger.info("Consumer closed")

    async def _process(self, handler: Callable[[Any], Awaitable[None]], msg: Any,) -> None:
        partition = msg.partition()
        lock = self._partition_locks.setdefault(partition, asyncio.Lock())
        try:
            async with lock:
                await handler(msg)
                if self._consumer:
                    await self._consumer.commit(msg)
        except Exception as e:
            logger.exception("Message processing failed %s", e)
        finally:
            self._semaphore.release()

    def stop(self) -> None:
        self._running = False



================================================
FILE: app/logging_config.py
================================================

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pythonjsonlogger.json import JsonFormatter


class _ServiceFormatter(JsonFormatter):
    def __init__(self, service_name: str, **kwargs):
        super().__init__(**kwargs)
        self._service = service_name

    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = (datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"))
        log_record["level"] = record.levelname
        log_record["service"] = self._service
        log_record["logger"] = record.name
        if record.exc_info and not log_record.get("exc_info"):
            log_record["exc_info"] = self.formatException(record.exc_info)

def setup_logging(service_name: str) -> None:

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = _ServiceFormatter(service_name)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Route uvicorn loggers through the same JSON handler so FastAPI
    # access logs are also structured.
    for uvicorn_logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(uvicorn_logger_name)
        uv_logger.handlers.clear()
        uv_logger.addHandler(handler)
        uv_logger.propagate = False



================================================
FILE: app/main.py
================================================
from __future__ import annotations

import asyncio
import logging
import signal
import httpx
from app.logging_config import setup_logging
from app.kafka_helpers import ReliableProducer, ConsumerLoop
from app.health import start_health_server
from app.config import settings
from app.consumer import process_event

setup_logging("kyc-consumer")
logger = logging.getLogger(__name__)


async def main() -> None:
    health_server = await start_health_server(settings.HEALTH_PORT)

    producer = ReliableProducer(settings.KAFKA_PRODUCER)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=settings.HTTP_MAX_CONNECTIONS,
            max_keepalive_connections=settings.HTTP_MAX_KEEPALIVE,
        ),
    )
    consumer_loop = ConsumerLoop(
        kafka_config=settings.KAFKA_CONSUMER,
        topics=[settings.MEDIA_KYC_CREATED],
        max_concurrency=settings.MAX_CONCURRENCY,
    )

    async def handler(msg):
        await process_event(msg, producer, http_client)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, consumer_loop.stop)
        except NotImplementedError:
            pass

    try:
        logger.info("KYC consumer starting…", extra={"max_concurrency": settings.MAX_CONCURRENCY})
        await consumer_loop.start(handler)
    finally:
        await http_client.aclose()
        await producer.flush()
        health_server.close()
        await health_server.wait_closed()
        logger.info("KYC consumer shut down")


if __name__ == "__main__":
    asyncio.run(main())



================================================
FILE: app/metrics.py
================================================
from __future__ import annotations
from prometheus_client import Counter, Histogram

# ── Consumer service metrics ──────────────────────────────────────────

messages_consumed_total = Counter(
    "messages_consumed_total",
    "Total Kafka messages consumed",
    labelnames=["service", "topic", "status"],
)

message_processing_duration_seconds = Histogram(
    "message_processing_duration_seconds",
    "End-to-end processing time per Kafka message",
    labelnames=["service", "topic"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)

http_client_request_duration_seconds = Histogram(
    "http_client_request_duration_seconds",
    "Duration of outbound HTTP calls to upstream services",
    labelnames=["service", "target_service", "endpoint", "status"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)

producer_publish_total = Counter(
    "producer_publish_total",
    "Total messages published to Kafka",
    labelnames=["service", "topic", "status"],
)



================================================
FILE: app/schema.py
================================================
import msgspec
from typing import Optional

class Storage(msgspec.Struct):
    provider: str
    bucket: str
    s3_key: str
    region: Optional[str] = None

# ---- KYC INPUT ----

class KYCData(msgspec.Struct):
    video_id: str
    user_id: str
    kyc_video_storage: Storage
    
class KYCEvent(msgspec.Struct):
    event_id: str
    timestamp: str
    data: KYCData

# ---- KYC OUTPUT ----

class KYCCompletedData(msgspec.Struct):
    event_id: str
    video_id: str
    user_id: str
    kyc_subtype: str
    status: str
    processing_time_ms: int

# ---- COMMON FAILURE OUTPUT ----

class ErrorInfo(msgspec.Struct):
    code: str
    message: str
    is_retryable: bool

class FailurePayload(msgspec.Struct):
    event_id: str
    video_id: str
    user_id: str
    s3_key: str
    status: str
    error: ErrorInfo



================================================
FILE: docker/docker-compose.yml
================================================
services:
  kyc-consumer:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    env_file:
      - ../.env
    environment:
      KAFKA_GROUP_ID: ai-kyc-processor
      FACE_SERVICE_URL: http://face-service:8001
      HEALTH_PORT: "8081"
    ports:
      - "8081:8081"
    restart: on-failure
    networks:
      - kafka-network

networks:
  kafka-network:
    external: true



================================================
FILE: docker/Dockerfile
================================================
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
CMD ["python", "-m", "app.main"]

"""


completion = client.chat.completions.create(
    model="deepseek-ai/deepseek-v4-pro",
    messages=[{"role":"user","content":PROMPT}],
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    extra_body={"chat_template_kwargs":{"thinking":False}},
    stream=True
)

for chunk in completion:
    if not getattr(chunk, "choices", None):
        continue
    if chunk.choices and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
