"""
Email Input Validator Module.

Provides pre-inference input validation to ensure submitted text or files
represent plausible email communications before invoking the frozen ML model.

Strictly independent of model weights, features, and decision thresholds.
Prevents arbitrary non-email text, junk, greetings alone, or generic paragraphs
from erroneously receiving HAM or SPAM classifications.
"""

import re
from typing import Dict, Any, Optional, Union

# Common RFC 822 / MIME Email Headers
HEADER_PATTERN = re.compile(
    r'^(?:From|To|Subject|Date|Cc|Bcc|Reply-To|Message-ID|Return-Path|'
    r'Content-Type|MIME-Version|Received|X-Mailer|Delivered-To|List-Unsubscribe):\s*.+',
    re.IGNORECASE | re.MULTILINE
)

# Common Email Salutations / Greetings at start of line or text
SALUTATION_PATTERN = re.compile(
    r'\b(?:hi|hello|hey|dear|good\s+(?:morning|afternoon|evening)|greetings|to\s+whom\s+it\s+may\s+concern|team|all|folks)\b',
    re.IGNORECASE
)

# Standalone Greeting Pattern (e.g. "Hi", "Hello Alex", "Dear Sir,") without substantive body
GREETING_ONLY_PATTERN = re.compile(
    r'^\s*(?:hi|hello|hey|dear|good\s+(?:morning|afternoon|evening)|greetings)\b(?:\s+[a-zA-Z\'-]+)?\s*[,.!?:;-]*\s*$',
    re.IGNORECASE
)

# Common Email Sign-offs / Closings / Acknowledgements
SIGNOFF_PATTERN = re.compile(
    r'\b(?:thanks|thank\s+you|regards|best\s+regards|kind\s+regards|warm\s+regards|'
    r'sincerely|best|cheers|yours|respectfully|talk\s+soon|take\s+care|warmly|'
    r'see\s+you|see\s+ya|catch\s+you\s+later|sounds\s+good|looks\s+good|works\s+for\s+me|'
    r'will\s+do|got\s+it|noted|looking\s+forward)\b',
    re.IGNORECASE
)

# Communicative, Action-Oriented, or Transactional Directives typical of emails
COMMUNICATIVE_DIRECTIVES = re.compile(
    r'\b(?:'
    # Request / Follow-up / Scheduling
    r'please|let\s+me\s+know|let\s+us\s+know|attached|attachment|see\s+attached|find\s+attached|'
    r'following\s+up|follow-up|can\s+(?:you|we)|could\s+(?:you|we)|would\s+(?:you|we)|'
    r'are\s+you\s+free|meet(?:ing)?|call\s+(?:me|us)|reply|respond|rsvp|schedule|feedback|'
    r'review|sync|reschedule|update|get\s+back\s+to\s+me|hope\s+you\s+are\s+well|'
    r'hope\s+this\s+finds\s+you\s+well|'
    # Security / Account / Phishing / Call-to-action
    r'click\s+here|click\s+below|click\s+the\s+link|verify\s+your|confirm\s+your|'
    r'log\s*in|sign\s*in|reset\s+password|security\s+alert|unauthorized|urgent|'
    r'action\s+required|suspend(?:ed)?|account\s+access|restore\s+access|'
    # Transactional / Marketing notices
    r'unsubscribe|newsletter|subscription|opt-out|privacy\s+policy|order\s*#?|'
    r'invoice\s*#?|tracking\s+number|receipt|payment|shipping'
    r')\b',
    re.IGNORECASE
)

# Email Address Pattern
EMAIL_ADDR_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# URL / Link Pattern
URL_PATTERN = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', re.IGNORECASE)

# HTML Markup Elements typical of rich emails
HTML_TAG_PATTERN = re.compile(r'<(?:html|body|p|a|div|table|tr|td|span|br|h[1-6])[\s/>]', re.IGNORECASE)


def validate_email_input(
    raw_payload: Union[str, bytes],
    parsed_email: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate whether submitted payload represents a plausible email or email body.

    Args:
        raw_payload: Raw string or bytes submitted by client.
        parsed_email: Optional dictionary produced by parse_raw_email.

    Returns:
        Dict containing:
            - is_valid (bool): True if suitable for ML classification.
            - error_code (str): Machine-readable code if invalid.
            - message (str): Clear plain-language user guidance.
            - details (dict): Diagnostic evidence (has_headers, word_count, signals).
    """
    # Step 1: Normalize payload to text
    if isinstance(raw_payload, bytes):
        text = raw_payload.decode("utf-8", errors="replace").strip()
    else:
        text = str(raw_payload or "").strip()

    # Step 2: Empty / Whitespace Check
    if not text:
        return {
            "is_valid": False,
            "error_code": "EMPTY_INPUT",
            "message": "Email payload is empty or contains only whitespace.",
            "details": {"raw_length": 0, "word_count": 0}
        }

    # Step 3: Extreme Degenerate / Non-text Filters
    raw_len = len(text)
    if raw_len < 10:
        return {
            "is_valid": False,
            "error_code": "INSUFFICIENT_CONTENT",
            "message": "Input is too short to be analyzed as an email (minimum 10 characters required).",
            "details": {"raw_length": raw_len, "word_count": len(text.split())}
        }

    # Standalone Greeting Filter (e.g. "Hi", "Hello Alex", "Dear Sir,")
    if GREETING_ONLY_PATTERN.match(text):
        return {
            "is_valid": False,
            "error_code": "INSUFFICIENT_CONTENT",
            "message": "Input contains only a greeting without any message body to analyze.",
            "details": {"raw_length": raw_len, "greeting_only": True}
        }

    # Punctuation & Letter Ratio Checks
    alpha_chars = [c for c in text if c.isalpha()]
    alpha_count = len(alpha_chars)
    if alpha_count < 3:
        return {
            "is_valid": False,
            "error_code": "INVALID_EMAIL_INPUT",
            "message": "Input contains no meaningful words or alphabetic text.",
            "details": {"raw_length": raw_len, "alpha_count": alpha_count}
        }

    # Repeated Character Check (e.g. "aaaaaaaaaaaaaa")
    char_counts = {}
    for c in text:
        if not c.isspace():
            char_counts[c] = char_counts.get(c, 0) + 1
    max_char_freq = max(char_counts.values()) if char_counts else 0
    non_space_len = sum(char_counts.values())
    if non_space_len > 12 and (max_char_freq / non_space_len) > 0.60:
        return {
            "is_valid": False,
            "error_code": "INVALID_EMAIL_INPUT",
            "message": "Input contains excessive repeated characters and does not resemble valid text.",
            "details": {"max_char_frequency_ratio": round(max_char_freq / non_space_len, 2)}
        }

    # Step 4: Token / Word Count Check
    words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", text)
    word_count = len(words)
    if word_count < 3:
        return {
            "is_valid": False,
            "error_code": "INSUFFICIENT_CONTENT",
            "message": "Input contains insufficient content for analysis (at least 3 words required).",
            "details": {"word_count": word_count}
        }

    # Step 5: Detect RFC 822 Email Headers
    header_matches = HEADER_PATTERN.findall(text)
    has_header_syntax = len(header_matches) > 0

    has_parsed_headers = False
    if parsed_email and isinstance(parsed_email, dict):
        has_parsed_headers = bool(
            (parsed_email.get("subject") and len(str(parsed_email.get("subject")).strip()) >= 2)
            or (parsed_email.get("from") and "@" in str(parsed_email.get("from")))
            or (parsed_email.get("to") and "@" in str(parsed_email.get("to")))
            or parsed_email.get("message_id")
            or parsed_email.get("date")
        )

    is_structured_email = has_header_syntax or has_parsed_headers

    # If it is a structured email with standard headers, accept it
    if is_structured_email:
        return {
            "is_valid": True,
            "error_code": "",
            "message": "",
            "details": {
                "format": "rfc822_structured",
                "header_count": len(header_matches),
                "word_count": word_count
            }
        }

    # Step 6: Body-Only Email Plausibility Evaluation
    # When no standard headers exist, verify that the text exhibits email communicative properties
    # rather than unrelated generic prose (e.g. encyclopedia articles, book excerpts).

    email_signals = []

    # 1. Salutation / Greeting
    if SALUTATION_PATTERN.search(text):
        email_signals.append("salutation")

    # 2. Sign-off / Closing
    if SIGNOFF_PATTERN.search(text):
        email_signals.append("signoff")

    # 3. Communicative Directives or Call-to-Action
    if COMMUNICATIVE_DIRECTIVES.search(text):
        email_signals.append("communicative_directive")

    # 4. Embedded Email Addresses
    if EMAIL_ADDR_PATTERN.search(text):
        email_signals.append("email_address")

    # 5. Embedded URLs / Links
    if URL_PATTERN.search(text) or (parsed_email and parsed_email.get("urls")):
        email_signals.append("urls_present")

    # 6. HTML Email Formatting
    if HTML_TAG_PATTERN.search(text) or (parsed_email and parsed_email.get("has_html")):
        email_signals.append("html_markup")

    # 7. Interpersonal Pronoun Address (2nd person + 1st person dialogue)
    lower_words = set(w.lower() for w in words)
    has_second_person = bool(lower_words & {"you", "your", "yours"})
    has_first_person = bool(lower_words & {"i", "me", "my", "we", "our", "us"})
    if has_second_person and has_first_person and (len(words) <= 40 or "?" in text):
        email_signals.append("interpersonal_address")

    # Acceptance threshold for body-only text:
    # Must exhibit at least one clear email communicative signal
    if len(email_signals) >= 1:
        return {
            "is_valid": True,
            "error_code": "",
            "message": "",
            "details": {
                "format": "body_only_plausible",
                "signals": email_signals,
                "word_count": word_count
            }
        }

    # Otherwise, reject generic unrelated paragraphs
    return {
        "is_valid": False,
        "error_code": "INVALID_EMAIL_INPUT",
        "message": (
            "Input does not appear to be an email message. Please enter a plausible email "
            "with standard headers (e.g. Subject:, From:) or a communicative message body "
            "(e.g. greeting, request, correspondence, or call-to-action)."
        ),
        "details": {
            "format": "unrelated_generic_text",
            "signals_detected": 0,
            "word_count": word_count
        }
    }
