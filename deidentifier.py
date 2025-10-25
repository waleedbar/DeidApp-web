# =================================================================
#  V10.3 - The Precision Engine (Surgical Regex Fixes)
#  Modified: Removed DetectedEntities and DetailedAnalysisReport columns
# =================================================================

import pandas as pd
import re
import logging
from typing import List, Dict, Set
import torch
import warnings
import os
import sys

# Fix for transformers compatibility issue
try:
    from functools import cached_property
    import transformers.utils as tf_utils
    if not hasattr(tf_utils, 'cached_property'):
        tf_utils.cached_property = cached_property
except Exception:
    pass

from transformers import pipeline

# Ensure all required packages are installed
try:
    from gliner import GLiNER
except ImportError:
    print("Installing required package: gliner")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gliner", "-q"])
    from gliner import GLiNER

# Install compatible versions if needed
try:
    import subprocess
    print("🔧 Checking and fixing package versions...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers>=4.30.0", "-q"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "tokenizers>=0.13.0", "-q"])
    print("✅ Package versions fixed!")
except Exception as e:
    print(f"⚠️ Warning during package update: {e}")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This part related to Google Colab is no longer needed on a server
# try:
#     from google.colab import drive
#     IN_COLAB = True
# except ImportError:
#     IN_COLAB = False

class AdvancedHybridDeidentifierV10_3:
    """
    V10.3 - The Precision Engine: Surgical fixes for greedy PHONE regex
    and enhanced ADDRESS detection for complex Arabic patterns.
    """
    def __init__(self, confidence_threshold: float = 0.85):
        logger.info("Initializing The Precision Engine V10.3...")
        self.confidence_threshold = confidence_threshold
        self.entity_priority = {
            'PHONE': 15, 'EMAIL': 15, 'ID': 14, 'DATE': 12, 'AGE': 11,
            'PERSON': 10, 'HCW': 9, 'PATIENT': 8, 'HOSP': 7, 'FACILITY': 7,
            'HOSPITAL': 7, 'CLINIC': 7, 'ADDRESS': 6
        }
        self.exclusion_list: Set[str] = self._build_exclusion_list()
        self.strict_rejection_list: Set[str] = self._build_strict_rejection_list()
        self.medical_terms: Set[str] = self._build_medical_terms_list()
        self.jordanian_hospitals_gazetteer = self._build_gazetteer_regex()
        self.known_locations: Set[str] = self._build_known_locations()
        self.models = {}
        self.load_models()

    def _build_exclusion_list(self) -> Set[str]:
        base_list = {'chest', 'heart', 'abdomen', 'abd', 'head', 'neck', 'back', 'limbs', 'examination', 'exam', 'assessment', 'plan', 'diagnosis', 'treatment', 'medication', 'medicine', 'drug', 'therapy', 'surgery', 'operation', 'vital', 'vitals', 'signs', 'temperature', 'pressure', 'pulse', 'rate', 'breathing', 'respiration', 'oxygen', 'saturation', 'blood', 'urine', 'lab', 'laboratory', 'test', 'result', 'report', 'imaging', 'xray', 'ultrasound', 'mri', 'ct', 'scan', 'ecg', 'eeg', 'normal', 'abnormal', 'pain', 'fever', 'headache', 'nausea', 'vomiting', 'diarrhea', 'cough', 'shortness', 'address', 'phone', 'county', 'marital', 'status', 'age', 'religion', 'sex', 'occupation', 'patient', 'history', 'examination', 'admission', 'assessment', 'plan', 'report', 'text', 'note', 'male', 'female', 'single', 'married', 'islam'}
        new_additions = {'المريضه', 'checkenpox', 'medlabs', 'contents', 'titer', 'enzymes', 'l.f.t', 'lft', 'liver function test', 'brucellaq', 'positive', 'gender', 'd.o.b', 'cyanosis', 'resp', 'distress', 'h-s megaly', 'ph', 'pco2', 'po2', 'o2 sat', 'hr', 'rr', 'abx', 'prbcs', 'pending', 'dm', 'htn', 'hypothyroid', 'thyroidectomy', 'tx', 'trx', 'fotr', 'fbs', 'll edema', 'feet ulcers', 'a1c', 'urinalyses', 'va', 'co', 'meningitis', 'patinet', 'prophylaxis', 'nkda', 'postive', 'trt', 'relation', 'prescriptions'}
        return base_list.union(new_additions)

    def _build_strict_rejection_list(self) -> Set[str]:
        """V10.3: Enhanced rejection list"""
        return {
            'emergency department', 'liver enzymes', 'enzymes', 'enzym', 'brucella titer',
            'checkenpox', 'l.f.t', 'positive', 'negative', 'medlabs', 'contents',
            'المريضه', 'available', 'avai lable', 'unknown', 'pt', 'doctor', 'doctors',
            'consult', 'see order', 'medical care', 'muscle strain', 's.o.a.p', 'soap', 'تقريرًا',
            'abdominal discomfort', 'abdominal pain', 'gastritis', 'functional pain',
            'epigastric pain', 'chest pain', 'back pain', 'joint pain', 'headache',
            'nausea', 'vomiting', 'diarrhea', 'constipation', 'dyspepsia',
            'internal medicine', 'family medicine', 'pediatrics', 'surgery',
            'orthopedics', 'cardiology', 'neurology', 'gastroenterology',
            'ultrasound abdomen', 'ultrasound', 'ct scan', 'mri', 'x-ray',
            'blood test', 'urine test', 'ecg', 'echo', 'endoscopy',
            'hepatosplenomegaly', 'cardiomegaly', 'lymphadenopathy',
            'organomegaly', 'edema', 'inflammation', 'infection',
            'university', 'hospital', 'clinic', 'center', 'building', 'street', 'st',
            'city', 'town', 'village', 'district', 'area', 'zone',
            'no known', 'not known', 'follow up', 'follow-up', 'advised',
            'discharge', 'admission', 'consultation', 'referral',
            'general eligibility', 'verified', 'rt tex', 'report text',
            'غير متوفر', 'غير-متوفر', 'not available', 'n/a'
        }

    def _build_medical_terms_list(self) -> Set[str]:
        """V10.3: Comprehensive medical terminology for validation"""
        return {
            'abdominal', 'gastric', 'cardiac', 'hepatic', 'renal', 'pulmonary',
            'neurological', 'dermatological', 'orthopedic', 'ophthalmic',
            'pain', 'discomfort', 'syndrome', 'disease', 'disorder', 'condition',
            'ultrasound', 'radiography', 'tomography', 'endoscopy', 'biopsy',
            'inflammation', 'infection', 'edema', 'hemorrhage', 'ischemia',
            'diagnosis', 'prognosis', 'treatment', 'therapy', 'medication'
        }

    def _build_known_locations(self) -> Set[str]:
        """V10.3: Known Jordanian locations for validation"""
        return {
            'عمان', 'الزرقاء', 'إربد', 'العقبة', 'المفرق', 'الكرك', 'معان', 'جرش',
            'amman', 'zarqa', 'irbid', 'aqaba', 'mafraq', 'karak', 'maan', 'jerash',
            'تلاع العلي', 'الجبيهة', 'الصويفية', 'الشميساني', 'عبدون', 'مرج الحمام',
            'الرابية', 'خلدا', 'الرصيفة', 'حكما', 'الطويل', 'المشاقبه', 'مضافه'
        }

    def _build_gazetteer_regex(self) -> str:
        names = ["مستشفى البشير", "Al-Basheer Hospital", "Prince Hamza Hospital", "Jordan University Hospital", "King Abdullah University Hospital", "Zarqa New Governmental Hospital", "Prince Faisal Hospital", "Istishari Hospital", "Specialty Hospital", "Jordan Hospital", "Arab Medical Center", "Istiklal Hospital", "Amman Comprehensive Health Center", "Jbaiha Comprehensive Clinic", "AIN AL-BASHA CENTER", "North Hashmi Clinic", "Yarmouk Governmental Hospital", "Ministry of Health", "Zarqa New Hospital", "P.H.H"]
        return r'\b(' + '|'.join(re.escape(name) for name in names) + r')\b'

    # ========================= START: IMPORTANT CHANGE HERE =========================
    def load_models(self):
        logger.info("🔄 [V10.3] Loading The Precision Engine Model Ensemble...")
        # Servers like Render (on the free plan) don't have a GPU, so we default to CPU
        device = -1 
        logger.info(f"💻 Using device: CPU")
        
        try:
            # This is the ID of your model on the Hugging Face Hub
            my_trained_model_path = "WaleedBR34/my-ultimate-ner-model" 
            
            logger.info(f"🔥 Loading custom-trained expert model from Hugging Face Hub: {my_trained_model_path}")
            # The pipeline library is smart enough to understand this ID
            # and will automatically download the model from the internet
            self.models['my_custom_ner_expert'] = pipeline("token-classification", model=my_trained_model_path, aggregation_strategy="simple", device=device)
            
            # The rest of the models are loaded as usual from Hugging Face
            logger.info("Loading general-purpose Arabic models...")
            self.models['gliner_arabic'] = GLiNER.from_pretrained("NAMAA-Space/gliner_arabic-v2.1")
            self.models['gliner_enhanced'] = GLiNER.from_pretrained("urchade/gliner_multi")
            logger.info("Loading English-based de-id models (as supplementary)...")
            self.models['roberta'] = pipeline("token-classification", model="obi/deid_roberta_i2b2", aggregation_strategy="simple", device=device)
            logger.info(f"🎯 {len(self.models)} models loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Critical error loading models: {e}", exc_info=True)
            raise
    # ========================== END: IMPORTANT CHANGE HERE ==========================

    def extract_entities_regex(self, text: str) -> List[Dict]:
        """
        V10.3 - THE PRECISION ENGINE: Surgically fixed regex patterns

        Key Improvements:
        1. PHONE patterns are now NON-GREEDY and precisely bounded
        2. New ADDRESS pattern specifically for Arabic slash-separated locations
        """
        entities = []
        patterns = {
            'HOSP': [self.jordanian_hospitals_gazetteer],
            'ID': [
                r'#\s*(\d{7,12})\b',
                r'\b(9\d{8,9})\b',
                r'(?:رقم وصل كشفيه|National ID|MRN|SSN)\s*:?\s*\(?#?\s*(\d{6,12})\s*\)?',
                r'\b\d{3}-\d{2}-\d{4}\b'
            ],
            # ========== V10.3 SURGICAL FIX: PHONE PATTERNS ==========
            'PHONE': [
                r'(?<![0-9])(?:\+?962|0)\s*7[789][\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d(?![0-9])',
                r'(?<![0-9])\+962\s?[6-7](?:[\s-]?\d){7,8}(?![0-9])'
            ],
            'DATE': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}(?:@\d{2}:\d{2})?'
            ],
            'AGE': [
                r'\b\d{1,2}yr\b',
                r'\b\d{1,2}yr\s+\d{1,2}mo\b',
                r'\b\d{1,2}\s+MONTHS?\s+OLD\b',
                r'\b\d{1,2}\s+YEARS?\s+OLD\b',
                r'\(\s*\d+\s*\)\s*yrs\b',
                r'\b\d{1,2}mo\b',
                r'\b\d{1,2}wk\b',
                r'\b\d{1,3}yrs\b'
            ],
            'HCW': [
                r'\bالدكتور\s+([\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){0,2})\b',
                r'\b(?:Dr|Doctor)\s*\.?\s*([A-Z][a-zA-Z\-]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-zA-Z\-]+){0,2})\b'
            ],
            'PERSON': [
                r'\b[\u0600-\u06FF]+,(?:[\u0600-\u06FF]+(?:[\s-]|-?الدين)?)+\b',
                r'\b(?:ابو|أم)-[\u0600-\u06FF]+(?:,\s?[\u0600-\u06FF]+)+\b'
            ],
            # ========== V10.3 ENHANCED: ADDRESS PATTERNS ==========
            'ADDRESS': [
                r'(?:Address|العنوان)\s*:?\s*([^\n]{10,100})',
                r'[\u0600-\u06FF\s]+-\s*[\u0600-\u06FF\s]+-\s*شارع\s+[\u0600-\u06FF\s]+-\s*عمارة\s+\d+',
                r'[\u0600-\u06FF]+/[\u0600-\u06FF]+-[\u0600-\u06FF\s,]+',
                r'\b(?:شارع|حي|منطقة|عمارة|بناية)\s+[\u0600-\u06FF\s]+(?:،|,)\s*(?:عمارة|بناية|رقم)?\s*\d+',
                r'\b[A-Za-z\u0600-\u06FF\s]+-\s+[A-Za-z\u0600-\u06FF\s]+(?:St\.?|Street)\s*-\s*(?:Bldg|Building)\s+\d+\b',
                r'\b(?:[A-Z][a-z]+\s*){2,}-\s+[A-Z][a-z]+\s+(?:St\.?|Street)\b',
                r'[\u0600-\u06FF]{2,}\s*[–-]\s*[\u0600-\u06FF]{2,}\s*[–-]\s*[\u0600-\u06FF\s]+',
                r'(?:شارع|St\.|Street)\s+[\u0600-\u06FFa-zA-Z\s]+(?:،|,)\s*(?:بناية|عمارة|Bldg\.?|Building)\s*\d+(?:،|,)?\s*[\u0600-\u06FFa-zA-Z\s]*',
                r'[\u0600-\u06FF]{2,}/[\u0600-\u06FF]{2,}(?:-[\u0600-\u06FF]{2,}){1,}(?:,\s*[\u0600-\u06FF]{2,})?'
            ],
            'EMAIL': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b']
        }

        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                try:
                    for match in re.finditer(pattern, text, re.IGNORECASE | re.UNICODE):
                        full_match_text = match.group(0).strip()
                        if not self._is_excluded(full_match_text):
                            entities.append({
                                'start': match.start(0),
                                'end': match.end(0),
                                'text': full_match_text,
                                'entity_type': entity_type,
                                'score': 1.0,
                                'model': 'regex'
                            })
                except re.error as e:
                    logger.warning(f"Regex error for pattern {pattern}: {e}")

        return entities

    def _is_excluded(self, text: str) -> bool:
        if not text or not text.strip():
            return True
        clean_text = text.strip().lower()
        if len(clean_text) <= 1:
            return True
        if clean_text.isdigit() and len(clean_text) <= 3:
            return True
        tokens = re.split(r'[\s/-]', clean_text)
        for token in tokens:
            if token in self.exclusion_list:
                return True
        return clean_text in self.exclusion_list

    def _is_valid_entity(self, entity: Dict) -> bool:
        text = entity['text'].strip()
        entity_type = entity['entity_type']
        if not text:
            return False
        if entity_type not in ['ID', 'PHONE', 'AGE', 'DATE'] and text.isdigit():
            return False
        if entity_type not in ['ID', 'HOSP'] and len(text) <= 4 and text.isupper() and text.isalpha():
            return False
        return True

    def _contains_medical_terminology(self, text: str) -> bool:
        """V10.3: Check if text contains medical terminology"""
        text_lower = text.lower()
        for term in self.medical_terms:
            if term in text_lower:
                return True
        return False

    def _is_single_generic_location_word(self, text: str) -> bool:
        """V10.3: Check if text is a single generic location word"""
        generic_words = {'university', 'hospital', 'clinic', 'center', 'building', 'street', 'st', 'city', 'town', 'village', 'district', 'area', 'zone', 'road', 'avenue'}
        return text.lower().strip() in generic_words and len(text.split()) == 1

    def _is_valid_email(self, text: str) -> bool:
        """V10.3: Validate email format"""
        return '@' in text and '.' in text.split('@')[-1] if '@' in text else False

    def extract_entities_from_models(self, text: str) -> List[Dict]:
        all_entities, gliner_labels = [], ["person", "doctor", "hospital", "clinic", "date", "age", "phone", "id", "address", "email"]
        for name, model in self.models.items():
            try:
                results = model.predict_entities(text, labels=gliner_labels, threshold=0.5) if 'gliner' in name else model(text)
                for r in results:
                    entity = ({'start': r['start'], 'end': r['end'], 'text': r['text'], 'entity_type': r['label'].upper(), 'score': r.get('score', r.get('confidence', 0.0))}
                              if 'gliner' in name else
                              {'start': r['start'], 'end': r['end'], 'text': r['word'], 'entity_type': r['entity_group'], 'score': r.get('score', 0.0)})
                    entity['text'], entity['model'] = entity['text'].strip(), name
                    all_entities.append(entity)
            except Exception as e:
                logger.error(f"Error with model {name}: {e}")
        return all_entities

    def merge_entities(self, all_entities: List[Dict], original_text: str) -> List[Dict]:
        if not all_entities:
            return []
        for entity in all_entities:
            entity['priority'] = self.entity_priority.get(entity['entity_type'], 0)
            if entity['model'] == 'regex':
                entity['priority'] += 100
            elif entity['model'] == 'my_custom_ner_expert':
                entity['priority'] += 75
            elif 'gliner' in entity['model']:
                entity['priority'] += 25
            else:
                entity['priority'] += 0
        entities = sorted(all_entities, key=lambda e: (e['start'], -e['end']))
        merged, i = [], 0
        while i < len(entities):
            current_cluster = [entities[i]]
            max_end, j = entities[i]['end'], i + 1
            while j < len(entities) and entities[j]['start'] < max_end:
                if entities[j]['end'] <= max_end:
                    current_cluster.append(entities[j])
                elif entities[j]['start'] < max_end:
                    current_cluster.append(entities[j])
                    max_end = max(max_end, entities[j]['end'])
                j += 1
            winner = max(current_cluster, key=lambda e: (e['priority'], e['score']))
            merged_entity = {
                'start': min(e['start'] for e in current_cluster),
                'end': max(e['end'] for e in current_cluster),
                'entity_type': winner['entity_type'],
                'score': winner['score'],
                'model': winner['model']
            }
            merged_entity['text'] = original_text[merged_entity['start']:merged_entity['end']]
            merged.append(merged_entity)
            i = j
        return merged

    def _post_process_and_validate(self, entities: List[Dict], original_text: str) -> List[Dict]:
        """V10.3: Final validation with all fortress rules"""
        validated_entities = []
        text_length = len(original_text)
        signature_zone_start = text_length * 0.85

        duration_keywords = {'for', 'x', 'since', 'ago', 'لمدة', 'منذ'}
        dosage_keywords = {'dose', 'tablet', 'bid', 'tid', 'qid', 'ml', 'mg', 'gm', 'mcg',
                           'جرعة', 'حبة', 'قرص', 'decrease', 'increase', 'give', 'given'}
        numeric_context_keywords = {'times', 'sib', 'sibling', 'abortion', 'gravida', 'para',
                                    'g', 'p', 'a', 'cs', 'nvd', 'pregnancies', 'deliveries'}

        for entity in entities:
            text_lower = entity['text'].lower().strip()

            if entity['entity_type'] in ['PERSON', 'HCW'] and entity['start'] > signature_zone_start:
                logger.info(f"🖋️ SIGNATURE identified (as {entity['entity_type']}): '{entity['text']}'")

            if text_lower in self.strict_rejection_list:
                logger.info(f"🛡️ [V10.3] REJECTED (Strict List): '{entity['text']}'")
                continue

            if entity['entity_type'] in ['PERSON', 'HCW', 'DOCTOR']:
                words = entity['text'].split()
                if len(words) > 1 and all(word.isupper() for word in words if word.isalpha()):
                    logger.info(f"🛡️ [V10.3] REJECTED (ALL CAPS Name Hallucination): '{entity['text']}'")
                    continue

            if entity['entity_type'] == 'ADDRESS':
                words = entity['text'].split()
                if len(words) == 1:
                    if text_lower not in self.known_locations and entity['text'] not in self.known_locations:
                        logger.info(f"🛡️ [V10.3] REJECTED (Single Word Address - Not Known Location): '{entity['text']}'")
                        continue

            if entity['entity_type'] == 'AGE':
                context_start = max(0, entity['start'] - 50)
                context_end = min(len(original_text), entity['end'] + 50)
                context_before = original_text[context_start:entity['start']].lower()
                context_after = original_text[entity['end']:context_end].lower()
                full_context = context_before + ' ' + context_after

                words_before = context_before.split()[-5:]
                if any(keyword in words_before for keyword in duration_keywords):
                    logger.info(f"🛡️ [V10.3] REJECTED (Context Rule - Duration): '{entity['text']}'")
                    continue

                words_context = full_context.split()
                if any(keyword in words_context for keyword in dosage_keywords):
                    logger.info(f"🛡️ [V10.3] REJECTED (Context Rule - Dosage): '{entity['text']}'")
                    continue

                if entity['text'].strip().isdigit() and len(entity['text'].strip()) <= 2:
                    if any(keyword in words_context for keyword in numeric_context_keywords):
                        logger.info(f"🛡️ [V10.3] REJECTED (Context Rule - Numeric Context): '{entity['text']}'")
                        continue

                if 'mg' in text_lower or 'mg' in full_context:
                    logger.info(f"🛡️ [V10.3] REJECTED (Context Rule - Dosage Unit): '{entity['text']}'")
                    continue

            if entity['entity_type'] in ['PERSON', 'HCW', 'PATIENT']:
                if self._contains_medical_terminology(entity['text']):
                    logger.info(f"🛡️ [V10.3] REJECTED (Medical Term in Name): '{entity['text']}'")
                    continue

            if entity['entity_type'] in ['LOC', 'ADDRESS', 'LOCATION']:
                if self._is_single_generic_location_word(entity['text']):
                    logger.info(f"🛡️ [V10.3] REJECTED (Generic Location Word): '{entity['text']}'")
                    continue

            if entity['entity_type'] == 'EMAIL':
                if not self._is_valid_email(entity['text']):
                    logger.info(f"🛡️ [V10.3] REJECTED (Invalid Email Format): '{entity['text']}'")
                    continue

            if entity['entity_type'] == 'AGE' and 'mg' in text_lower:
                logger.info(f"🛡️ [V10.3] REJECTED (Logical Rule): '{entity['text']}'")
                continue

            if entity['entity_type'] in ['HOSP', 'CLINIC', 'HOSPITAL'] and text_lower in ['hospital', 'clinic', 'center', 'private clinic', 'our hospital']:
                logger.info(f"🛡️ [V10.3] REJECTED (Generic Facility): '{entity['text']}'")
                continue

            if entity['entity_type'] in ['PERSON', 'HCW', 'DOCTOR'] and len(entity['text'].split()) == 1 and not any('\u0600' <= char <= '\u06FF' for char in entity['text']) and not entity['text'][0].isupper() and entity['model'] != 'regex':
                logger.info(f"🛡️ [V10.3] REJECTED (Invalid Name): '{entity['text']}'")
                continue

            validated_entities.append(entity)

        return validated_entities

    def apply_entities_to_text(self, text: str, entities: List[Dict]) -> str:
        if not entities:
            return text
        parts, last_end = [], 0
        for entity in sorted(entities, key=lambda x: x['start']):
            parts.append(text[last_end:entity['start']])
            parts.append(f"[{entity['entity_type']}]")
            last_end = entity['end']
        parts.append(text[last_end:])
        return self._clean_result("".join(parts))

    def _clean_result(self, text: str) -> str:
        text = re.sub(r'(?<=\S) +(?=\S)', ' ', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()

    def format_entities_for_display(self, entities: List[Dict]) -> str:
        if not entities:
            return "No entities detected"
        return " | ".join([f"'{e['text']}' ({e['entity_type']}/{e.get('model', 'N/A')}) - Score: {e.get('score', 0):.2f}" for e in entities])

    def create_detailed_analysis_report(self, original_text: str, result: str, entities: List[Dict]) -> str:
        normalized_result = re.sub(r'\[(HOSP|FACILITY|HOSPITAL|CLINIC)\]', '[HOSP]', result)
        report_lines = [
            f"--- ⚙️ Detailed Analysis Report (V10.3 - The Precision Engine) ---",
            "",
            "📄 Original Text:",
            original_text,
            "",
            "✅ De-identified Result:",
            normalized_result,
            ""
        ]
        if entities:
            report_lines.append("🎯 Detected Entities:")
            for i, entity in enumerate(entities, 1):
                report_lines.append(f"  {i}. '{entity['text']}' → [{entity['entity_type']}] (Model: {entity.get('model', 'N/A')}, Score: {entity.get('score', 0):.2f})")
        else:
            report_lines.append("🎯 Detected Entities: No entities detected")
        report_lines.extend(["", "📊 Summary:", f"  • Total entities detected: {len(entities)}"])
        if entities:
            entity_types = {}
            for entity in entities:
                entity_type = re.sub(r'(HOSP|FACILITY|HOSPITAL|CLINIC)', 'HOSP', entity['entity_type'])
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            report_lines.append(f"  • Entity types found: {', '.join([f'{k}({v})' for k, v in entity_types.items()])}")
        report_lines.extend(["", "-" * 50])
        return "\n".join(report_lines)

    def process_single_text(self, text: str) -> Dict:
        if not isinstance(text, str) or not text.strip():
            return {'final_merged_result': text, 'final_merged_entities': [], 'detailed_report': 'Empty or invalid text'}
        regex_entities = self.extract_entities_regex(text)
        model_entities = self.extract_entities_from_models(text)
        all_entities = regex_entities + model_entities
        filtered_entities = [e for e in all_entities if (e.get('score', 0) >= self.confidence_threshold and not self._is_excluded(e['text']) and self._is_valid_entity(e))]
        merged_entities = self.merge_entities(filtered_entities, text)
        final_entities = self._post_process_and_validate(merged_entities, text)
        result_text = self.apply_entities_to_text(text, final_entities)
        report = self.create_detailed_analysis_report(text, result_text, final_entities)
        return {'final_merged_result': result_text, 'final_merged_entities': final_entities, 'detailed_report': report}

    def process_excel_file(self, input_file: str, output_file: str):
        try:
            logger.info(f"📁 Loading Excel file: {input_file}")
            df = pd.read_excel(input_file)
            if 'OrigionalText' not in df.columns:
                raise ValueError("Input file must contain a column named 'OrigionalText'")
            logger.info(f"📊 Processing {len(df)} rows with The Precision Engine V10.3...")
            results = df['OrigionalText'].astype(str).apply(self.process_single_text)

            # ========== MODIFIED: Only save DeidentifiedText column ==========
            df['DeidentifiedText_V10_3'] = [r.get('final_merged_result', row) for r, row in zip(results, df['OrigionalText'])]
            # Removed: DetectedEntities_V10_3 and DetailedAnalysisReport_V10_3 columns

            logger.info(f"💾 Saving de-identified results to: {output_file}")
            df.to_excel(output_file, index=False, engine='openpyxl')
            logger.info("✅ Excel processing complete with V10.3 Precision Engine!")
        except FileNotFoundError:
            logger.error(f"❌ File not found: {input_file}")
        except Exception as e:
            logger.error(f"❌ A critical error occurred during Excel processing: {e}")

# This part is removed because the file will be run by app.py, not directly
# def main():
#     ...

# if __name__ == "__main__":
#     main()