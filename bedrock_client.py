import json
import logging
import time
import boto3
from botocore.exceptions import ClientError
from dataclasses import dataclass
from typing import Optional

# Configure logger for this module
logger = logging.getLogger(__name__)

MODEL_ID = "amazon.nova-micro-v1:0"

SYSTEM_PROMPT = """You are a mental clarity assistant. Help people stop overthinking.

Analyze their thought and respond with ONE action:
- DO_IT_NOW: urgent, in their control, quick (under 15 mins)
- SCHEDULE_IT: important but needs time, or not urgent
- LET_IT_GO: outside their control, or just anxiety with no actionable step

Respond ONLY with this JSON:
{"action": "DO_IT_NOW|SCHEDULE_IT|LET_IT_GO", "summary": "1-sentence of what they're really saying", "reason": "why this action", "next_step": "one concrete thing to do"}"""

ACTION_MAP = {
    "DO_IT_NOW": {"emoji": "✅", "title": "Do it now"},
    "SCHEDULE_IT": {"emoji": "🕒", "title": "Schedule it"},
    "LET_IT_GO": {"emoji": "🗑", "title": "Let it go"}
}


@dataclass
class TokenMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0

    def log(self):
        logger.info("┌─────────────────────────────────────")
        logger.info("│ TOKEN METRICS")
        logger.info("├─────────────────────────────────────")
        logger.info(f"│ Input tokens:    {self.input_tokens:>6}")
        logger.info(f"│ Output tokens:   {self.output_tokens:>6}")
        logger.info(f"│ Total tokens:    {self.total_tokens:>6}")
        logger.info(f"│ Latency:         {self.latency_ms:>6.0f} ms")
        if self.latency_ms > 0:
            tokens_per_sec = (self.output_tokens / self.latency_ms) * 1000
            logger.info(f"│ Output speed:    {tokens_per_sec:>6.1f} tok/s")
        logger.info("└─────────────────────────────────────")


class BedrockClient:
    def __init__(self, region_name: str = "us-east-1"):
        logger.info(f"Initializing Bedrock client | Region: {region_name} | Model: {MODEL_ID}")
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = MODEL_ID
        self._total_requests = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def analyze_thought(self, thought: str) -> dict:
        self._total_requests += 1
        request_id = self._total_requests

        logger.info("=" * 50)
        logger.info(f"REQUEST #{request_id}")
        logger.info("=" * 50)
        logger.info(f"Input: {thought[:100]}{'...' if len(thought) > 100 else ''}")
        logger.info(f"Input length: {len(thought)} chars")

        try:
            return self._call_bedrock(thought, request_id)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            logger.error(f"Bedrock API Error | Code: {error_code}")
            logger.error(f"Message: {error_msg}")
            return self._fallback_response()
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return self._fallback_response()

    def _call_bedrock(self, thought: str, request_id: int) -> dict:
        request_params = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": thought}]}],
            "system": [{"text": SYSTEM_PROMPT}],
            "inferenceConfig": {"maxTokens": 300, "temperature": 0.7}
        }

        logger.debug(f"Request config: maxTokens=300, temperature=0.7")

        start_time = time.time()
        response = self.client.converse(**request_params)
        latency_ms = (time.time() - start_time) * 1000

        # Extract token metrics
        metrics = TokenMetrics(latency_ms=latency_ms)

        if "usage" in response:
            usage = response["usage"]
            metrics.input_tokens = usage.get("inputTokens", 0)
            metrics.output_tokens = usage.get("outputTokens", 0)
            metrics.total_tokens = metrics.input_tokens + metrics.output_tokens

            # Update session totals
            self._total_input_tokens += metrics.input_tokens
            self._total_output_tokens += metrics.output_tokens

        # Log token metrics
        metrics.log()

        # Log session totals
        logger.info(f"Session totals: {self._total_requests} requests | "
                   f"{self._total_input_tokens} in | {self._total_output_tokens} out")

        # Extract and parse result
        result_text = response["output"]["message"]["content"][0]["text"]
        logger.info(f"Raw output ({len(result_text)} chars):")
        logger.info(f"  {result_text}")

        # Log stop reason if available
        if "stopReason" in response:
            logger.debug(f"Stop reason: {response['stopReason']}")

        parsed_result = self._parse_response(result_text)
        parsed_result["tokens_used"] = metrics.total_tokens
        logger.info(f"Parsed action: {parsed_result['action'].upper()}")
        logger.info("=" * 50)

        return parsed_result

    def _parse_response(self, result_text: str) -> dict:
        start = result_text.find("{")
        end = result_text.rfind("}") + 1

        if start == -1 or end <= start:
            logger.warning("No JSON found in response")
            raise ValueError("No JSON found in response")

        json_str = result_text[start:end]
        result = json.loads(json_str)

        action = result.get("action", "LET_IT_GO").upper().replace(" ", "_")
        if action not in ACTION_MAP:
            logger.warning(f"Unknown action '{action}', defaulting to LET_IT_GO")
            action = "LET_IT_GO"

        return {
            "action": action.lower(),
            "emoji": ACTION_MAP[action]["emoji"],
            "title": ACTION_MAP[action]["title"],
            "summary": result.get("summary", ""),
            "reason": result.get("reason", ""),
            "next_step": result.get("next_step", "")
        }

    def _fallback_response(self) -> dict:
        logger.info("Returning fallback response")
        return {
            "action": "let_it_go",
            "emoji": "🗑",
            "title": "Let it go",
            "summary": "Having trouble analyzing this thought.",
            "reason": "When in doubt, release it.",
            "next_step": "Take a breath and try again.",
            "tokens_used": 0
        }

    def get_session_stats(self) -> dict:
        """Get session statistics for monitoring."""
        return {
            "total_requests": self._total_requests,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens
        }
