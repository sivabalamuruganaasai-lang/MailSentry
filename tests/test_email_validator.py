"""
Unit Tests for Pre-Inference Email Input Validator (src/data/email_validator.py).

Verifies that the validator correctly discriminates between valid email payloads
(structured emails or plausible email bodies) and non-email inputs (empty text,
whitespace, greetings alone, random short text, repeated characters, generic paragraphs).

Guarantees:
- Pure pre-inference structural and communicative validation.
- Zero dependency on model weights, feature matrices, or decision thresholds.
- Rejection of invalid inputs without returning HAM or SPAM classifications.
"""

import pytest
from src.data.email_validator import validate_email_input
from src.data.parser import parse_raw_email


class TestEmailInputValidator:
    """Test suite for validate_email_input function."""

    # 1. Empty and Whitespace-Only Inputs
    def test_empty_string_rejected(self):
        res = validate_email_input("")
        assert res["is_valid"] is False
        assert res["error_code"] == "EMPTY_INPUT"
        assert "empty" in res["message"].lower()

    def test_whitespace_only_rejected(self):
        res = validate_email_input("   \n\t  \r\n   ")
        assert res["is_valid"] is False
        assert res["error_code"] == "EMPTY_INPUT"

    def test_none_input_rejected(self):
        res = validate_email_input(None)
        assert res["is_valid"] is False
        assert res["error_code"] == "EMPTY_INPUT"

    def test_empty_bytes_rejected(self):
        res = validate_email_input(b"")
        assert res["is_valid"] is False
        assert res["error_code"] == "EMPTY_INPUT"

    # 2. Extremely Short / Greetings Alone
    def test_random_short_text_rejected(self):
        res = validate_email_input("abc")
        assert res["is_valid"] is False
        assert res["error_code"] == "INSUFFICIENT_CONTENT"

    def test_hello_world_rejected(self):
        res = validate_email_input("hello world")
        assert res["is_valid"] is False
        assert res["error_code"] == "INSUFFICIENT_CONTENT"
        assert "greeting" in res["message"].lower() or "insufficient" in res["message"].lower()

    def test_single_greeting_alone_rejected(self):
        greetings = [
            "Hi",
            "Hello",
            "Hey",
            "Dear John,",
            "Good morning!",
            "Greetings,",
            "Good evening everyone."
        ]
        for g in greetings:
            res = validate_email_input(g)
            assert res["is_valid"] is False, f"Expected greeting '{g}' to be rejected"
            assert res["error_code"] == "INSUFFICIENT_CONTENT"

    # 3. Punctuation and Degenerate Text
    def test_punctuation_only_rejected(self):
        puncts = [
            "!!!!!!!!??????......",
            "---...---...---",
            ";;;;;;;;;;;;;;;;;",
            "$%^&*()@#$%"
        ]
        for p in puncts:
            res = validate_email_input(p)
            assert res["is_valid"] is False
            assert res["error_code"] in ["INVALID_EMAIL_INPUT", "INSUFFICIENT_CONTENT"]

    def test_repeated_characters_rejected(self):
        repeated = [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "xxxxxxxxxxxxxxxxxxxx",
            "11111111111111111111"
        ]
        for r in repeated:
            res = validate_email_input(r)
            assert res["is_valid"] is False
            assert res["error_code"] in ["INVALID_EMAIL_INPUT", "INSUFFICIENT_CONTENT"]

    # 4. Unrelated Generic Paragraphs (Encyclopedia, Fiction, Articles)
    def test_photosynthesis_paragraph_rejected(self):
        text = (
            "Photosynthesis is a process used by plants and other organisms to convert light "
            "energy into chemical energy that, through cellular respiration, can later be released "
            "to fuel the organism's metabolic activities. This chemical energy is stored in "
            "carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water."
        )
        res = validate_email_input(text)
        assert res["is_valid"] is False
        assert res["error_code"] == "INVALID_EMAIL_INPUT"
        assert "email message" in res["message"].lower()

    def test_historical_encyclopedia_text_rejected(self):
        text = (
            "The French Revolution was a period of political and societal change in France that "
            "began with the Estates General of 1789 and ended with the formation of the French Consulate "
            "in November 1799. Many of its ideas are considered fundamental principles of Western liberal democracy."
        )
        res = validate_email_input(text)
        assert res["is_valid"] is False
        assert res["error_code"] == "INVALID_EMAIL_INPUT"

    def test_fiction_excerpt_rejected(self):
        text = (
            "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, "
            "his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through "
            "the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust "
            "from entering along with him."
        )
        res = validate_email_input(text)
        assert res["is_valid"] is False
        assert res["error_code"] == "INVALID_EMAIL_INPUT"

    # 5. Plausible Short Emails (Accepted)
    def test_plausible_short_emails_accepted(self):
        short_emails = [
            "Sounds good, see you tomorrow at the office.",
            "Thanks for the update, talk soon!",
            "Got it, will review the document by end of day.",
            "Can we reschedule our sync to 3pm tomorrow?",
            "Are you free for a quick call this afternoon?",
            "Hi Sarah, could you please send the updated numbers? Thanks, Mike",
            "Please find the signed agreement attached. Best regards, Jane"
        ]
        for email in short_emails:
            res = validate_email_input(email)
            assert res["is_valid"] is True, f"Expected short email '{email}' to be accepted, got: {res}"

    # 6. Structured RFC 822 Emails (Accepted)
    def test_structured_rfc822_email_with_headers_accepted(self):
        rfc_email = (
            "From: alex@corp.com\n"
            "To: team@corp.com\n"
            "Subject: Quarterly Roadmap Review\n"
            "Date: Mon, 12 Oct 2026 09:00:00 +0000\n\n"
            "Please review the attached roadmap slides before our discussion."
        )
        res = validate_email_input(rfc_email)
        assert res["is_valid"] is True
        assert res["details"]["format"] == "rfc822_structured"

    def test_partial_headers_email_accepted(self):
        # Email with only Subject
        subject_only = "Subject: Quick question about tomorrow\n\nAre you available around 2pm?"
        res1 = validate_email_input(subject_only)
        assert res1["is_valid"] is True

        # Email with only From
        from_only = "From: hr@company.com\n\nPlease submit your timesheet before Friday."
        res2 = validate_email_input(from_only)
        assert res2["is_valid"] is True

    # 7. Phishing / Security Alert Email (Accepted for Analysis, Verdict NOT Hard-Coded)
    def test_phishing_alert_email_accepted_for_analysis(self):
        phish_text = (
            "Urgent: Your account access has been suspended due to unauthorized activity. "
            "Please click here to verify your identity and restore access: "
            "http://192.168.1.1/login-verify.html"
        )
        res = validate_email_input(phish_text)
        assert res["is_valid"] is True, "Phishing email must be accepted for ML analysis"
        # Crucial: Validation result must NOT contain a model verdict
        assert "is_spam" not in res
        assert "label" not in res
        assert "routing_tier" not in res

    # 8. HTML Rich Emails (Accepted)
    def test_html_email_accepted(self):
        html_email = (
            "<html><body>"
            "<h2>Your Order Confirmation</h2>"
            "<p>Thank you for your recent purchase. Tracking number: 987654321.</p>"
            "<p><a href='https://store.example.com/track'>Track Package</a></p>"
            "</body></html>"
        )
        res = validate_email_input(html_email)
        assert res["is_valid"] is True
        assert "html_markup" in res["details"].get("signals", [])

    # 9. Integration with parse_raw_email
    def test_integration_with_parsed_email(self):
        raw_bytes = (
            b"From: billing@cloud.com\r\n"
            b"To: admin@company.com\r\n"
            b"Subject: Monthly Cloud Invoice\r\n\r\n"
            b"Your cloud services invoice for October is now available."
        )
        parsed = parse_raw_email(raw_bytes)
        res = validate_email_input(raw_bytes, parsed_email=parsed)
        assert res["is_valid"] is True
        assert res["details"]["format"] == "rfc822_structured"
