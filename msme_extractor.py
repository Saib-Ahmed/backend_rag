"""
MSME Form Extractor Module
==========================
Refactored from pdf-extractor v4 for use as a FastAPI-integrated module.
Extracts structured MSME legal form fields from PDFs, images, and text files
using Gemini (primary) and NVIDIA Kimi K2.6 (fallback) LLMs.

Manages per-session extraction state to support incremental extraction
across multiple document uploads.
"""

import os
import json
import time
import shutil
import logging
import requests
import base64
import re
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

# ==========================================
# Configuration
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Directory that holds the Hindi form template
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILE = os.path.join(_MODULE_DIR, "form_template.md")

# Root directory for session state files
SESSIONS_DIR = os.path.join(_MODULE_DIR, "msme_sessions")

# All 70+ form fields
TARGET_KEYS = [
    "applicant_name", "respondent_name", "authorized_representative_name",
    "authorized_representative_firm_name", "buyer_name_for_material_or_service",
    "udyog_aadhaar_number", "udyam_registration_number", "registration_document_1_page_number",
    "application_submission_date", "authorized_representative_document_number",
    "authorized_representative_page_number", "aggrieved_mse_unit_name",
    "aggrieved_mse_address_with_pincode", "aggrieved_mse_state", "aggrieved_mse_district",
    "aggrieved_mse_mobile_number", "aggrieved_mse_email", "aggrieved_mse_type_micro_or_small",
    "respondent_buyer_name", "respondent_buyer_address_with_pincode", "respondent_buyer_state",
    "respondent_buyer_district", "respondent_buyer_mobile_number", "respondent_buyer_email",
    "respondent_buyer_category_cpsu_state_psu_other", "supply_orders_details_document_number",
    "supply_orders_details_page_number", "bills_invoices_details_document_number",
    "bills_invoices_details_page_number", "delivery_challan_or_completion_certificate_document_number",
    "delivery_challan_or_completion_certificate_page_number", "acknowledgement_of_material_document_number",
    "acknowledgement_of_material_page_number", "agreement_or_contract_document_number",
    "agreement_or_contract_page_number", "payable_and_unpaid_invoices_document_number",
    "payable_and_unpaid_invoices_page_number", "delayed_payment_days_calculation_details",
    "correspondence_with_respondent_document_number", "correspondence_with_respondent_page_number",
    "respondent_complaints_or_objections_details", "rectification_and_acceptance_proof_details",
    "acceptance_after_rectification_details", "principal_amount_due", "interest_amount_claimed",
    "interest_period_from_date", "interest_period_to_date", "table_invoice_or_bill_number_and_date",
    "table_total_invoice_or_bill_amount", "table_material_receipt_date", "table_amount_received_with_details",
    "table_payment_date", "table_delay_period", "table_outstanding_principal_amount",
    "table_interest_claim_bill_number", "table_due_date_45_days", "table_total_days_of_delay",
    "table_bank_interest_rate", "table_compound_interest_calculation", "table_remarks_document_or_page",
    "index_doc1_udyam_registration_page_range", "index_doc2_authorization_letter_page_range",
    "index_doc3_contract_agreement_page_range", "index_doc4_supply_order_page_range",
    "index_doc5_bills_invoices_page_range", "index_doc6_delivery_challan_page_range",
    "index_doc7_work_completion_certificate_page_range",
    "index_doc8_payment_correspondence_legal_notice_page_range",
    "index_doc9_compound_interest_calculation_sheet_page_range",
    "index_doc10_bank_statement_ledger_page_range",
    "index_doc11_audited_balance_sheet_page_range",
    "index_doc12_notarized_affidavit_page_range",
    "index_doc13_other_gst_certificate_form_3_and_3b_page_range", "relief_sought",
    "witness_1_name_and_address", "witness_2_name_and_address",
    "case_summary_or_other_relevant_info",
    "additional_audited_balance_sheet_ledger_document_number",
    "additional_audited_balance_sheet_ledger_page_number",
    "additional_dtic_memorandum_document_number", "additional_dtic_memorandum_page_number",
    "additional_affidavit_document_number", "additional_affidavit_page_number",
    "additional_other_documents_gst_certificate_document_number",
    "additional_other_documents_gst_certificate_page_number",
    "form_signing_date", "form_signature", "form_signer_name",
    "form_signer_designation", "form_enterprise_seal",
]


class MsmeExtractor:
    """Per-session MSME form field extractor with incremental state management."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = os.path.join(SESSIONS_DIR, session_id)
        self.state_file = os.path.join(self.session_dir, "state.json")
        os.makedirs(self.session_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {key: "" for key in TARGET_KEYS}

    def _save_state(self, state: dict) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4, ensure_ascii=False)

    def get_state(self) -> dict:
        return self._load_state()

    def get_progress(self) -> dict:
        state = self._load_state()
        filled = sum(1 for v in state.values() if str(v).strip())
        total = len(TARGET_KEYS)
        missing = [k for k, v in state.items() if not str(v).strip()]
        return {
            "total_fields": total,
            "filled_fields": filled,
            "missing_fields_count": total - filled,
            "missing_fields": missing,
            "percent_complete": round((filled / total) * 100, 1) if total else 0,
        }

    def reset(self) -> None:
        """Clear all extraction state for this session."""
        if os.path.exists(self.session_dir):
            shutil.rmtree(self.session_dir)
        os.makedirs(self.session_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _fill_template(template: str, data: dict) -> str:
        def _replace(match):
            field_name = match.group(1).strip()
            return str(data.get(field_name, ""))
        return re.sub(r"\{\{(.*?)\}\}", _replace, template)

    def get_filled_form(self) -> str:
        if not os.path.exists(TEMPLATE_FILE):
            return "⚠️ Template file not found."
        with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
            template = f.read()
        state = self._load_state()
        return self._fill_template(template, state)

    # ------------------------------------------------------------------
    # LLM extraction — Gemini (primary)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_with_gemini(file_bytes: bytes, filename: str, mime_type: str, missing_keys: list) -> dict:
        """Send document bytes to Gemini API and extract missing fields."""
        if not GEMINI_API_KEY or not GEMINI_API_KEY.strip() or "placeholder" in GEMINI_API_KEY.lower():
            raise ValueError("GEMINI_API_KEY is not set or is empty in the server environment variables.")
        logger.info(f"Gemini extraction for {len(missing_keys)} missing fields from '{filename}'...")
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Upload the file bytes as a temporary file for the Gemini Files API
        import tempfile
        ext = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        uploaded_file = None
        extracted_data = {}
        try:
            uploaded_file = client.files.upload(file=tmp_path)
            time.sleep(2)

            # Split missing_keys into batches of 20 to prevent schema state space explosion
            batch_size = 20
            for i in range(0, len(missing_keys), batch_size):
                batch_keys = missing_keys[i : i + batch_size]
                logger.info(f"Extracting Gemini batch {i // batch_size + 1} ({len(batch_keys)} fields)...")

                dynamic_schema = {
                    "type": "OBJECT",
                    "properties": {key: {"type": "STRING"} for key in batch_keys},
                }

                prompt = (
                    "Look at this document directly and read out the values for the "
                    "requested missing fields. Output them strictly structured into the "
                    "schema. If a field is missing, output an empty string."
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt, uploaded_file],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=dynamic_schema,
                        temperature=0.0,
                    ),
                )
                try:
                    batch_res = json.loads(response.text)
                    if isinstance(batch_res, dict):
                        extracted_data.update(batch_res)
                except Exception as e:
                    logger.error(f"Failed to parse batch json: {e}. Text: {response.text}")

            return extracted_data
        finally:
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception as e:
                    logger.error(f"Failed to delete uploaded Gemini file: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # LLM extraction — NVIDIA Kimi K2.6 (fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_with_kimi(file_bytes: bytes, filename: str, mime_type: str, missing_keys: list) -> dict:
        """Fallback extraction using Moonshot Kimi-K2.6 via NVIDIA."""
        if not NVIDIA_API_KEY or not NVIDIA_API_KEY.strip() or "placeholder" in NVIDIA_API_KEY.lower():
            raise ValueError("NVIDIA_API_KEY is not set or is empty in the server environment variables.")
        logger.info(f"Kimi fallback extraction for {len(missing_keys)} fields from '{filename}'...")
        invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
        }

        dynamic_target = {key: "" for key in missing_keys}
        prompt_text = (
            f"Analyze the document and extract values for these specific missing fields. "
            f"Map them to this JSON template: {json.dumps(dynamic_target)}. "
            f"Do not return fields outside this template."
        )

        content_list = [{"type": "text", "text": prompt_text}]
        ext = os.path.splitext(filename)[1].lower()

        # Text files — send as inline text
        if ext in (".md", ".txt"):
            document_text = file_bytes.decode("utf-8", errors="replace")
            content_list[0]["text"] += f"\n\nHere is the document text:\n{document_text}"

        # Images — send as base64
        elif ext in (".png", ".jpg", ".jpeg"):
            b64_img = base64.b64encode(file_bytes).decode("utf-8")
            img_mime = "image/png" if ext == ".png" else "image/jpeg"
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img_mime};base64,{b64_img}"},
            })

        # PDFs — render each page to JPEG via PyMuPDF
        else:
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("jpeg")
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                })

        payload = {
            "model": "moonshotai/kimi-k2.6",
            "messages": [{"role": "user", "content": content_list}],
            "max_tokens": 16384,
            "temperature": 0.1,
            "top_p": 1.00,
            "stream": False,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse output JSON robustly
        try:
            return json.loads(content.strip())
        except Exception:
            # Fallback to extracting JSON block between markdown tags or curly braces
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except Exception:
                    pass
            # Try finding the first '{' and last '}'
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start:end+1].strip())
                except Exception:
                    pass
            raise ValueError(f"Kimi returned text that could not be parsed as JSON: {content[:500]}...")

    # ------------------------------------------------------------------
    # Main extraction pipeline
    # ------------------------------------------------------------------
    def extract(self, file_bytes: bytes, filename: str, mime_type: str) -> dict:
        """
        Process a single file and incrementally update extraction state.
        Returns a result dict with status, fields updated, and progress.
        """
        current_state = self._load_state()
        missing_keys = [k for k, v in current_state.items() if not str(v).strip()]

        if not missing_keys:
            return {
                "status": "complete",
                "message": "All fields are already filled!",
                "fields_updated": 0,
                **self.get_progress(),
            }

        # Attempt extraction with Gemini first, fallback to Kimi
        new_data = None
        provider_used = None
        try:
            new_data = self._extract_with_gemini(file_bytes, filename, mime_type, missing_keys)
            provider_used = "gemini"
            logger.info("✅ Gemini extraction succeeded.")
        except Exception as e:
            logger.warning(f"⚠️ Gemini failed: {e}")
            try:
                new_data = self._extract_with_kimi(file_bytes, filename, mime_type, missing_keys)
                provider_used = "kimi"
                logger.info("✅ Kimi extraction succeeded.")
            except Exception as fallback_e:
                logger.error(f"❌ Both providers failed. Kimi error: {fallback_e}")
                return {
                    "status": "error",
                    "message": f"Both extraction providers failed. Gemini: {e}, Kimi: {fallback_e}",
                    "fields_updated": 0,
                    **self.get_progress(),
                }

        # Merge new data into state
        fields_updated = 0
        if new_data:
            for key in missing_keys:
                extracted_value = str(new_data.get(key, "")).strip()
                if extracted_value:
                    current_state[key] = extracted_value
                    fields_updated += 1
            self._save_state(current_state)

        progress = self.get_progress()
        return {
            "status": "success",
            "message": f"Extracted {fields_updated} new fields using {provider_used}.",
            "fields_updated": fields_updated,
            "provider": provider_used,
            **progress,
        }

    def extract_from_text(self, text: str) -> dict:
        """Extract fields from plain text (e.g., voice transcript or chat message)."""
        text_bytes = text.encode("utf-8")
        return self.extract(text_bytes, "user_input.txt", "text/plain")
