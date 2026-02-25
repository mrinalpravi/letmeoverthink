import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger(__name__)

# Configuration
REQUESTS_PER_MINUTE = 10
TOKENS_PER_HOUR = 5000
MAX_INPUT_LENGTH = 2000  # characters


@dataclass
class UserUsage:
    request_timestamps: list = field(default_factory=list)
    token_usage: list = field(default_factory=list)  # (timestamp, tokens)
    total_requests: int = 0
    total_tokens: int = 0


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int = REQUESTS_PER_MINUTE,
        tokens_per_hour: int = TOKENS_PER_HOUR,
        max_input_length: int = MAX_INPUT_LENGTH
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_hour = tokens_per_hour
        self.max_input_length = max_input_length
        self._usage: dict[str, UserUsage] = defaultdict(UserUsage)
        self._lock = Lock()

        logger.info(
            f"Rate limiter initialized | "
            f"Requests: {requests_per_minute}/min | "
            f"Tokens: {tokens_per_hour}/hour | "
            f"Max input: {max_input_length} chars"
        )

    def _get_client_ip(self) -> str:
        """Get client IP, handling proxies."""
        if request.headers.get("X-Forwarded-For"):
            return request.headers.get("X-Forwarded-For").split(",")[0].strip()
        return request.remote_addr or "unknown"

    def _cleanup_old_entries(self, usage: UserUsage, now: float):
        """Remove expired timestamps."""
        minute_ago = now - 60
        hour_ago = now - 3600

        usage.request_timestamps = [
            ts for ts in usage.request_timestamps if ts > minute_ago
        ]
        usage.token_usage = [
            (ts, tokens) for ts, tokens in usage.token_usage if ts > hour_ago
        ]

    def check_request_limit(self) -> tuple[bool, str | None]:
        """Check if request is within rate limits."""
        client_ip = self._get_client_ip()
        now = time.time()

        with self._lock:
            usage = self._usage[client_ip]
            self._cleanup_old_entries(usage, now)

            # Check requests per minute
            if len(usage.request_timestamps) >= self.requests_per_minute:
                oldest = min(usage.request_timestamps)
                wait_time = int(60 - (now - oldest)) + 1
                logger.warning(
                    f"Rate limit exceeded | IP: {client_ip} | "
                    f"Requests: {len(usage.request_timestamps)}/{self.requests_per_minute}/min"
                )
                return False, f"Too many requests. Please wait {wait_time} seconds."

            # Check tokens per hour
            tokens_used = sum(tokens for _, tokens in usage.token_usage)
            if tokens_used >= self.tokens_per_hour:
                logger.warning(
                    f"Token limit exceeded | IP: {client_ip} | "
                    f"Tokens: {tokens_used}/{self.tokens_per_hour}/hour"
                )
                return False, "Token limit exceeded. Please try again later."

            return True, None

    def check_input_length(self, text: str) -> tuple[bool, str | None]:
        """Check if input is within length limits."""
        if len(text) > self.max_input_length:
            logger.warning(
                f"Input too long | Length: {len(text)} | Max: {self.max_input_length}"
            )
            return False, f"Input too long. Maximum {self.max_input_length} characters allowed."
        return True, None

    def record_request(self):
        """Record a request for the current client."""
        client_ip = self._get_client_ip()
        now = time.time()

        with self._lock:
            usage = self._usage[client_ip]
            usage.request_timestamps.append(now)
            usage.total_requests += 1

            logger.debug(
                f"Request recorded | IP: {client_ip} | "
                f"Requests this minute: {len(usage.request_timestamps)}"
            )

    def record_tokens(self, tokens: int):
        """Record token usage for the current client."""
        client_ip = self._get_client_ip()
        now = time.time()

        with self._lock:
            usage = self._usage[client_ip]
            usage.token_usage.append((now, tokens))
            usage.total_tokens += tokens

            tokens_this_hour = sum(t for _, t in usage.token_usage)
            logger.info(
                f"Tokens recorded | IP: {client_ip} | "
                f"This request: {tokens} | "
                f"This hour: {tokens_this_hour}/{self.tokens_per_hour}"
            )

    def get_usage_stats(self, client_ip: str = None) -> dict:
        """Get usage stats for a client or all clients."""
        with self._lock:
            if client_ip:
                usage = self._usage.get(client_ip)
                if not usage:
                    return {}
                now = time.time()
                self._cleanup_old_entries(usage, now)
                return {
                    "requests_this_minute": len(usage.request_timestamps),
                    "tokens_this_hour": sum(t for _, t in usage.token_usage),
                    "total_requests": usage.total_requests,
                    "total_tokens": usage.total_tokens,
                }
            else:
                return {
                    "total_clients": len(self._usage),
                    "total_requests": sum(u.total_requests for u in self._usage.values()),
                    "total_tokens": sum(u.total_tokens for u in self._usage.values()),
                }


# Global instance
rate_limiter = RateLimiter()


def require_rate_limit(f):
    """Decorator to enforce rate limiting on routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check rate limit
        allowed, error_msg = rate_limiter.check_request_limit()
        if not allowed:
            return jsonify({"error": error_msg, "rate_limited": True}), 429

        # Check input length if POST with JSON
        if request.method == "POST" and request.is_json:
            data = request.get_json(silent=True) or {}
            thought = data.get("thought", "")
            allowed, error_msg = rate_limiter.check_input_length(thought)
            if not allowed:
                return jsonify({"error": error_msg}), 400

        # Record the request
        rate_limiter.record_request()

        return f(*args, **kwargs)

    return decorated_function
