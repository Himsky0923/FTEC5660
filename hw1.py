#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel
    from langchain_deepseek import ChatDeepSeek
    from pydantic import BaseModel, Field
    import os

    llm = ChatDeepSeek(model="deepseek-v4-flash-vision-exp", api_key=os.getenv("DEEPSEEK_API_KEY"))

    class Items(BaseModel):
        original_prices: list[float] = Field(
            description=(
                "One number per item line, including a $0.00 item line such as COUPON or VCODE. "
                "Use the dollar amount printed on the right of that line. "
                "If QTY or 數量 is shown and the line already has a total, do not multiply. "
                "Exclude discount lines, ROUNDING, 小計, SUBTOTAL, the payment line, 找續, CHANGE, 餘額, points, and card numbers."
            )
        )
        quantities: list[int] = Field(
            description=(
                "Same length and order as original_prices. "
                "The QTY or 數量 printed on that item line. Use 1 when that line has no quantity."
            )
        )
        discounts: list[float] = Field(
            default_factory=list,
            description=(
                "One positive number per discount line: promotion, coupon amount, member, app, "
                "packaging damage, Buy-X-Save, and percentage off. "
                "A discount that covers several items is still one number. "
                "Do not include a $0.00 item line. Do not include ROUNDING."
            ),
        )

    class Footer(BaseModel):
        item_count: int = Field(
            description="The integer printed on the left of 小計 or SUBTOTAL. Copy that integer. Do not count rows."
        )
        subtotal: float = Field(
            description="The amount printed on the 小計 or SUBTOTAL line. Copy it. Do not calculate it."
        )
        rounding: float = Field(
            default=0.0,
            description="The ROUNDING line exactly as printed, usually negative. Use 0 if that line is absent.",
        )
        paid: float = Field(
            description=(
                "The amount on the payment line directly under ROUNDING, such as OCTOPUS or VISA, after rounding. "
                "Not 小計, not SUBTOTAL, not CHANGE, not 找續, not 餘額."
            )
        )

    items_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Return JSON for the item and discount lines of this one supermarket receipt. "
            "Keys: original_prices (array of numbers), quantities (array of integers), discounts (array of numbers). "
            "original_prices: one number per item line, including a $0.00 item line such as COUPON or VCODE. "
            "Use the dollar amount printed on the right of that line. "
            "If QTY or 數量 is shown and the line already has a total, do not multiply. "
            "Exclude discount lines, ROUNDING, 小計, SUBTOTAL, the payment line, 找續, CHANGE, 餘額, points, and card numbers. "
            "quantities: same length and order as original_prices. "
            "Use the QTY or 數量 on that item line, or 1 when it is not printed. "
            "discounts: one positive number per discount line "
            "(promotion, coupon amount, member, app, packaging damage, Buy-X-Save, percentage off). "
            "A discount that covers several items is still one number. "
            "Do not include a $0.00 item line. Do not include ROUNDING. "
            "Labels may be Chinese or English. Copy printed amounts only. Do not add or subtract.",
        ),
        (
            "human",
            [
                {"type": "text", "text": "Extract the item lines and the discount lines."},
                {"type": "image_url", "image_url": {"url": "{image_url}"}},
            ],
        ),
    ])

    footer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Return JSON for the totals block of this one supermarket receipt. "
            "Keys: item_count (integer), subtotal (number), rounding (number), paid (number). "
            "Copy the printed totals. Do not list items and do not calculate. "
            "item_count: the integer on the left of 小計 or SUBTOTAL. "
            "subtotal: the amount on the 小計 or SUBTOTAL line. "
            "rounding: the ROUNDING line exactly as printed, usually negative. Use 0 if that line is absent. "
            "paid: the amount on the payment line directly under ROUNDING, such as OCTOPUS or VISA. "
            "Not 小計, not SUBTOTAL, not CHANGE, not 找續, not 餘額.",
        ),
        (
            "human",
            [
                {"type": "text", "text": "Read the totals block only."},
                {"type": "image_url", "image_url": {"url": "{image_url}"}},
            ],
        ),
    ])

    items_chain = items_prompt | llm.with_structured_output(Items, method="json_mode")
    footer_chain = footer_prompt | llm.with_structured_output(Footer, method="json_mode")
    return RunnableParallel(items=items_chain, footer=footer_chain)


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    def money(value: Any) -> Decimal:
        return Decimal(str(value)).quantize(Decimal("0.01"))

    def parts(result: Any) -> tuple[Decimal, Decimal, bool]:
        items = result["items"]
        footer = result["footer"]
        prices = [money(price) for price in items.original_prices]
        quantities = [int(qty) for qty in items.quantities]
        discounts = [money(amount) for amount in items.discounts]
        subtotal = money(footer.subtotal)
        rounding = money(footer.rounding)
        paid = money(footer.paid)
        price_sum = sum(prices, Decimal("0.00"))
        discount_sum = sum(discounts, Decimal("0.00"))
        # The dollar identity alone can pass when an item and a discount of the
        # same amount are both missing. The printed unit count catches that.
        # Subtotal and paid are read in a separate call, so the item call cannot
        # edit them to force a match.
        matched = (
            len(prices) == len(quantities)
            and sum(quantities) == int(footer.item_count)
            and price_sum - discount_sum == subtotal
            and subtotal + rounding == paid
        )
        return paid, price_sum, matched

    pending = list(images)
    chosen: dict[Path, tuple[Decimal, Decimal]] = {}
    for attempt in range(0, 3):
        if not pending:
            break
        results = chain.batch(
            [{"image_url": image_data_url(path)} for path in pending]
        )
        failed: list[Path] = []
        for path, result in zip(pending, results):
            paid, price_sum, matched = parts(result)
            print(f"try {attempt + 1} {path.name}: paid={paid} prices={price_sum} match={matched}")
            chosen[path] = (paid, price_sum)
            if not matched:
                failed.append(path)
        pending = failed

    spent = Decimal("0.00")
    without_discount = Decimal("0.00")
    for path in images:
        paid, price_sum = chosen[path]
        spent += paid
        without_discount += price_sum

    return {
        QUERY_1: f"HK${spent:.2f}",
        QUERY_2: f"HK${without_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())