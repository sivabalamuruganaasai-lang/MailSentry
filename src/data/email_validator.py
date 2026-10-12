"""
Email Input Validator Module.

Provides robust pre-inference input validation to ensure submitted text or files
represent plausible email communications before invoking the frozen ML model.

Strictly independent of model weights, features, and decision thresholds.
Prevents arbitrary non-email text, junk, greetings alone, or generic paragraphs
from erroneously receiving HAM or SPAM classifications.

Validation Pipeline Architecture:
Stage 1: Input presence, size, and degenerate text filters (empty, length < 10, repeated chars, greeting only).
Stage 2: Deterministic RFC 822 header parsing & value validation from message preamble.
Stage 3: Body extraction and contextual content assessment (handling missing/empty subjects vs. meaningful bodies).
Stage 4: Body-only plausibility evaluation for headerless pasted text (greetings, directives, URLs, sign-offs).
"""

import re
from typing import Dict, Any, Optional, Union, List, Tuple

# Recognized standard RFC 822 / MIME Email Headers
STANDARD_EMAIL_HEADERS = {
    "from", "to", "subject", "date", "cc", "bcc", "reply-to", "message-id",
    "return-path", "content-type", "mime-version", "received", "x-mailer",
    "delivered-to", "list-unsubscribe", "resent-from", "resent-to", "resent-date"
}

# Regex for standard RFC 822 header line: starts with token followed by colon
HEADER_LINE_PATTERN = re.compile(r'^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$')

# Common email salutations / greetings at start of line or text
SALUTATION_PATTERN = re.compile(
    r'\b(?:hi|hello|hey|dear|good\s+(?:morning|afternoon|evening)|greetings|to\s+whom\s+it\s+may\s+concern|team|all|folks)\b',
    re.IGNORECASE
)

# Standalone greeting pattern (e.g. "Hi", "Hello Alex", "Dear Sir,") without substantive body
GREETING_ONLY_PATTERN = re.compile(
    r'^\s*(?:hi|hello|hey|dear|good\s+(?:morning|afternoon|evening)|greetings)\b(?:\s+[a-zA-Z\'-]+)?\s*[,.!?:;-]*\s*$',
    re.IGNORECASE
)

# Common email sign-offs / closings / acknowledgements
SIGNOFF_PATTERN = re.compile(
    r'\b(?:thanks|thank\s+you|regards|best\s+regards|kind\s+regards|warm\s+regards|'
    r'sincerely|best|cheers|yours|respectfully|talk\s+soon|take\s+care|warmly|'
    r'see\s+you|see\s+ya|catch\s+you\s+later|sounds\s+good|looks\s+good|works\s+for\s+me|'
    r'will\s+do|got\s+it|noted|looking\s+forward)\b',
    re.IGNORECASE
)

# Communicative, action-oriented, transactional, or security directives typical of emails
COMMUNICATIVE_DIRECTIVES = re.compile(
    r'\b(?:'
    # Request / Follow-up / Scheduling / Meeting
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

# Email address pattern
EMAIL_ADDR_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# URL pattern
URL_PATTERN = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', re.IGNORECASE)

# HTML markup elements typical of rich emails
HTML_TAG_PATTERN = re.compile(r'<(?:html|body|p|a|div|table|tr|td|span|br|h[1-6])[\s/>]', re.IGNORECASE)


def detect_email_communicative_signals(
    text: str,
    parsed_email: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Extract communicative and semantic indicators typical of email communications."""
    signals = []

    # 1. Salutation / Greeting
    if SALUTATION_PATTERN.search(text):
        signals.append("salutation")

    # 2. Sign-off / Closing / Conversational acknowledgment
    if SIGNOFF_PATTERN.search(text):
        signals.append("signoff")

    # 3. Communicative Directives or Call-to-Action
    if COMMUNICATIVE_DIRECTIVES.search(text):
        signals.append("communicative_directive")

    # 4. Embedded Email Addresses
    if EMAIL_ADDR_PATTERN.search(text):
        signals.append("email_address")

    # 5. Embedded URLs / Links
    if URL_PATTERN.search(text) or (parsed_email and parsed_email.get("urls")):
        signals.append("urls_present")

    # 6. HTML Email Formatting
    if HTML_TAG_PATTERN.search(text) or (parsed_email and parsed_email.get("has_html")):
        signals.append("html_markup")

    # 7. Interpersonal Pronoun Address in Conversational Context
    words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", text)
    lower_words = set(w.lower() for w in words)
    has_second_person = bool(lower_words & {"you", "your", "yours"})
    has_first_person = bool(lower_words & {"i", "me", "my", "we", "our", "us"})
    if has_second_person and has_first_person and (len(words) <= 40 or "?" in text):
        signals.append("interpersonal_address")

    return signals


def parse_rfc822_header_preamble(lines: List[str]) -> Tuple[Dict[str, str], int, bool]:
    """
    Deterministically parses an RFC 822 header block starting strictly at line 0.

    Returns:
        headers (Dict[str, str]): Lowercased header names mapped to normalized values.
        body_start_line (int): Index where body content begins.
        is_preamble (bool): True if line 0 began a valid RFC 822 header block.
    """
    if not lines:
        return {}, 0, False

    # A genuine RFC 822 header block MUST start at line 0.
    # Header-like patterns embedded inside ordinary prose do NOT constitute a preamble.
    first_line = lines[0].strip()
    m_first = HEADER_LINE_PATTERN.match(first_line)
    if not m_first:
        return {}, 0, False

    first_hdr_name = m_first.group(1).lower()
    if first_hdr_name not in STANDARD_EMAIL_HEADERS:
        return {}, 0, False

    headers: Dict[str, str] = {}
    curr_header: Optional[str] = None
    body_start_line = len(lines)

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Blank line signals standard RFC 822 header/body separator (\r\n\r\n or \n\n)
        if not stripped:
            body_start_line = i + 1
            break

        # Folded header line (starts with space or tab per RFC 822)
        if curr_header and (line.startswith(" ") or line.startswith("\t")):
            headers[curr_header] += " " + stripped
            continue

        m = HEADER_LINE_PATTERN.match(stripped)
        if m:
            hdr_name = m.group(1).lower()
            hdr_val = m.group(2).strip()
            curr_header = hdr_name
            if hdr_name in headers:
                headers[hdr_name] += ", " + hdr_val
            else:
                headers[hdr_name] = hdr_val
            continue

        # Encountered a non-header line before any blank line separator
        # This line marks the transition to body
        body_start_line = i
        break

    has_standard = any(h in headers for h in STANDARD_EMAIL_HEADERS)
    return headers, body_start_line, has_standard


def validate_email_input(
    raw_payload: Union[str, bytes],
    parsed_email: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate whether submitted payload represents a plausible email or email body.

    Executes a 4-stage deterministic validation pipeline before model invocation:
    Stage 1: Input Presence, Minimum Content & Degenerate Text Checks.
    Stage 2: RFC 822 Header Preamble Extraction & Value Integrity Checks.
    Stage 3: Body Content Assessment & Contextual Validation.
    Stage 4: Body-Only Communicative Plausibility Assessment.

    Args:
        raw_payload: Raw string or bytes submitted by client.
        parsed_email: Optional dictionary produced by parse_raw_email.

    Returns:
        Dict containing:
            - is_valid (bool): True if suitable for ML classification.
            - error_code (str): Machine-readable code if invalid.
            - message (str): Clear user guidance.
            - details (dict): Diagnostic evidence.
    """
    # -------------------------------------------------------------------------
    # Stage 1: Input Presence, Length & Degenerate Text Checks
    # -------------------------------------------------------------------------
    if isinstance(raw_payload, bytes):
        text = raw_payload.decode("utf-8", errors="replace").strip()
    else:
        text = str(raw_payload or "").strip()

    # Empty / whitespace check
    if not text:
        return {
            "is_valid": False,
            "error_code": "EMPTY_INPUT",
            "message": "Email payload is empty or contains only whitespace.",
            "details": {"raw_length": 0, "word_count": 0}
        }

    raw_len = len(text)
    if raw_len < 10:
        return {
            "is_valid": False,
            "error_code": "INSUFFICIENT_CONTENT",
            "message": "Input is too short to be analyzed as an email (minimum 10 characters required).",
            "details": {"raw_length": raw_len, "word_count": len(text.split())}
        }

    # Standalone greeting check (e.g. "Hi", "Hello Alex", "Dear Sir,")
    if GREETING_ONLY_PATTERN.match(text):
        return {
            "is_valid": False,
            "error_code": "INSUFFICIENT_CONTENT",
            "message": "Input contains only a greeting without any message body to analyze.",
            "details": {"raw_length": raw_len, "greeting_only": True}
        }

    # Alphabetic character count check
    alpha_chars = [c for c in text if c.isalpha()]
    alpha_count = len(alpha_chars)
    if alpha_count < 3:
        return {
            "is_valid": False,
            "error_code": "INVALID_EMAIL_INPUT",
            "message": "Input contains no meaningful words or alphabetic text.",
            "details": {"raw_length": raw_len, "alpha_count": alpha_count}
        }

    # Repeated character ratio check (e.g. "aaaaaaaaaaaaaa")
    char_counts: Dict[str, int] = {}
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

    all_words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", text)
    total_word_count = len(all_words)
    if total_word_count < 3:
        return {
            "is_valid": False,
            "error_code": "INSUFFICIENT_CONTENT",
            "message": "Input contains insufficient content for analysis (at least 3 words required).",
            "details": {"word_count": total_word_count}
        }

    # -------------------------------------------------------------------------
    # Stage 2: RFC 822 Header Preamble Extraction & Value Integrity Checks
    # -------------------------------------------------------------------------
    lines = text.splitlines()
    headers, body_start_idx, has_header_preamble = parse_rfc822_header_preamble(lines)

    if has_header_preamble:
        # Validate required header values: From and To must have non-empty, meaningful content
        if "from" in headers:
            from_val = headers["from"].strip()
            if not from_val or not any(c.isalnum() for c in from_val):
                return {
                    "is_valid": False,
                    "error_code": "INVALID_EMAIL_INPUT",
                    "message": "Email contains malformed or empty required header values (From: is empty or invalid).",
                    "details": {"malformed_header": "from"}
                }

        if "to" in headers:
            to_val = headers["to"].strip()
            if not to_val or not any(c.isalnum() for c in to_val):
                return {
                    "is_valid": False,
                    "error_code": "INVALID_EMAIL_INPUT",
                    "message": "Email contains malformed or empty required header values (To: is empty or invalid).",
                    "details": {"malformed_header": "to"}
                }

        # ---------------------------------------------------------------------
        # Stage 3: Body Extraction & Contextual Content Assessment
        # ---------------------------------------------------------------------
        body_text = "\n".join(lines[body_start_idx:]).strip()
        body_words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", body_text)
        body_word_count = len(body_words)

        subject_val = headers.get("subject", "").strip()
        subject_words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", subject_val)
        subject_word_count = len(subject_words)
        has_meaningful_subject = subject_word_count >= 2

        # Case 3A: No subject (or empty subject) AND no body
        # E.g. isolated From: and To: lines with no body
        if subject_word_count == 0 and body_word_count == 0:
            return {
                "is_valid": False,
                "error_code": "INSUFFICIENT_CONTENT",
                "message": "Email has no subject and no message body to analyze.",
                "details": {
                    "format": "rfc822_empty_subject_and_body",
                    "headers_present": list(headers.keys())
                }
            }

        # Case 3B: No subject (or empty subject) WITH body
        # When subject is empty/missing, the body MUST be assessed for genuine email communicative plausibility.
        # From and To alone with an empty subject and an unrelated paragraph (e.g. Wikipedia) MUST be rejected!
        if subject_word_count == 0 and body_word_count > 0:
            if body_word_count < 3:
                return {
                    "is_valid": False,
                    "error_code": "INSUFFICIENT_CONTENT",
                    "message": "Email has an empty subject and insufficient body content to analyze (at least 3 words required).",
                    "details": {"body_word_count": body_word_count}
                }

            # Check if body exhibits email communicative characteristics
            body_signals = detect_email_communicative_signals(body_text, parsed_email)
            if len(body_signals) >= 1:
                return {
                    "is_valid": True,
                    "error_code": "",
                    "message": "",
                    "details": {
                        "format": "rfc822_no_subject_valid_body",
                        "signals": body_signals,
                        "body_word_count": body_word_count
                    }
                }
            else:
                return {
                    "is_valid": False,
                    "error_code": "INVALID_EMAIL_INPUT",
                    "message": (
                        "Email has no subject and the message body does not exhibit email "
                        "communicative characteristics (e.g. greeting, request, scheduling, or call-to-action)."
                    ),
                    "details": {
                        "format": "rfc822_no_subject_unrelated_body",
                        "body_word_count": body_word_count,
                        "signals_detected": 0
                    }
                }

        # Case 3C: Subject is present and meaningful
        if has_meaningful_subject:
            # End-of-Message (EOM) email with subject only
            if body_word_count == 0:
                if subject_word_count >= 3:
                    return {
                        "is_valid": True,
                        "error_code": "",
                        "message": "",
                        "details": {
                            "format": "rfc822_eom_subject_only",
                            "subject_word_count": subject_word_count
                        }
                    }
                else:
                    return {
                        "is_valid": False,
                        "error_code": "INSUFFICIENT_CONTENT",
                        "message": "Input contains insufficient subject or body content for analysis.",
                        "details": {"subject_word_count": subject_word_count}
                    }

            # Standard email with headers, subject, and body
            return {
                "is_valid": True,
                "error_code": "",
                "message": "",
                "details": {
                    "format": "rfc822_structured",
                    "header_count": len(headers),
                    "total_word_count": body_word_count + subject_word_count
                }
            }

        # Single word subject with empty body
        if subject_word_count == 1 and body_word_count == 0:
            return {
                "is_valid": False,
                "error_code": "INSUFFICIENT_CONTENT",
                "message": "Input contains insufficient subject or body content for analysis.",
                "details": {"subject_word_count": subject_word_count}
            }

    # -------------------------------------------------------------------------
    # Stage 4: Body-Only Input Assessment (Headerless Pasted Text or Prose)
    # -------------------------------------------------------------------------
    # When no standard header block is present at line 0, evaluate whether the text
    # represents a plausible email message body (greeting, request, signoff, URL, etc.)
    # rather than unrelated generic prose, literature excerpts, or article paragraphs.
    signals = detect_email_communicative_signals(text, parsed_email)

    if len(signals) >= 1:
        return {
            "is_valid": True,
            "error_code": "",
            "message": "",
            "details": {
                "format": "body_only_plausible",
                "signals": signals,
                "word_count": total_word_count
            }
        }

    # Reject unrelated generic paragraphs or prose with zero email signals
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
            "word_count": total_word_count
        }
    }
