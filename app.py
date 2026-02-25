import logging
from flask import Flask, render_template, request, jsonify
from bedrock_client import BedrockClient
from rate_limiter import rate_limiter, require_rate_limit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Set DEBUG level for bedrock_client to see all details
logging.getLogger("bedrock_client").setLevel(logging.DEBUG)

# Reduce noise from other libraries
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = Flask(__name__)
bedrock = BedrockClient()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
@require_rate_limit
def analyze():
    data = request.json
    thought = data.get("thought", "").strip()

    if not thought:
        logger.warning("Empty thought received")
        return jsonify({"error": "No thought provided"}), 400

    logger.info(f"Received thought for analysis: {thought[:50]}...")
    result = bedrock.analyze_thought(thought)

    # Record token usage for rate limiting
    tokens_used = result.get("tokens_used", 0)
    if tokens_used > 0:
        rate_limiter.record_tokens(tokens_used)

    logger.info(f"Returning result with action: {result['action']}")
    return jsonify(result)


@app.route("/stats")
def stats():
    """Endpoint to check usage stats (for monitoring)."""
    return jsonify(rate_limiter.get_usage_stats())


if __name__ == "__main__":
    logger.info("Starting Let Me Overthink server...")
    app.run(host="0.0.0.0",debug=True, port=5001)
