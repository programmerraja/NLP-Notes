import spacy
from typing import List
from rapidfuzz import fuzz
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
import json
# 1. THE ADVANCED NUMERIC RECOGNIZER
class ClinicalNumericRecognizer(EntityRecognizer):
    def __init__(self, nlp_model):
        # We pass the nlp model directly to avoid the nlp_artifacts error
        super().__init__(supported_entities=["PHONE_NUMBER", "ID"], name="ClinicalNumericRecognizer")
        self.nlp = nlp_model

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None):
        results = []
        # We re-process or use the doc to check 'like_num'
        doc = self.nlp(text)
        
        current_span = []
        for token in doc:
            # Matches '300', 'three hundred', '29', 'twenty-nine', '0', 'oh'
            if token.like_num or token.text.lower() in ["oh", "zero"]:
                current_span.append(token)
            else:
                # If we have a sequence of 2+ numeric tokens (for DOB/Address/ID)
                # or a very long one for Phone
                if len(current_span) >= 2:
                    start = current_span[0].idx
                    end = current_span[-1].idx + len(current_span[-1].text)
                    
                    # Heuristic: If it's 7+ tokens, it's likely a phone/ID
                    # If it's 2-3, it's part of an address or DOB
                    label = "PHONE_NUMBER" if len(current_span) >= 5 else "ID"
                    
                    results.append(RecognizerResult(
                        entity_type=label,
                        start=start,
                        end=end,
                        score=0.95
                    ))
                current_span = []
        return results

# 2. THE STATE MANAGER (Consistency)
class PIIStateManager:
    def __init__(self):
        self.entity_map = {}
        self.counters = {"PERSON": 1, "PHONE_NUMBER": 1, "DATE_TIME": 1, "ID": 1}
        self.type_map = {"PERSON": "NAME", "PHONE_NUMBER": "PHONE", "DATE_TIME": "DOB", "ID": "ID"}

    def get_replacement(self, text: str, label: str) -> str:
        norm_text = text.lower().strip()
        if norm_text in self.entity_map:
            return self.entity_map[norm_text]
        
        tag = self.type_map.get(label, "ID")
        count = self.counters.get(label, 1)
        placeholder = f"[{tag}_{count}]"
        self.entity_map[norm_text] = placeholder
        self.counters[label] += 1
        return placeholder

# 3. INTEGRATED PIPELINE
class MedicalRedactor:
    def __init__(self):
        # Load Transformer
        model_name = "en_core_web_trf"
        self.nlp = spacy.load(model_name)
        
        # Configure Presidio
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        
        # Add the Smart Numeric Recognizer
        self.analyzer.registry.add_recognizer(ClinicalNumericRecognizer(self.nlp))
        self.state = PIIStateManager()

    def redact(self, turns: List[str]) -> List[str]:
        output = []
        for text in turns:
            results = self.analyzer.analyze(text=text, language="en", 
                                            entities=["PERSON", "PHONE_NUMBER", "DATE_TIME", "ID"])
            
            # Sort reverse to prevent index shifting
            results = sorted(results, key=lambda x: x.start, reverse=True)
            print(text)
            redacted_text = text
            for res in results:
                original = text[res.start:res.end]
                replacement = self.state.get_replacement(original, res.entity_type)
                redacted_text = redacted_text[:res.start] + replacement + redacted_text[res.end:]
            output.append(redacted_text)
        return output

# --- TEST ---
if __name__ == "__main__":
    transcript = [
        "Patient is John Doe.",
        "My ID is three hundred twenty nine zero zero.", # Complex verbal numbers
        "My phone is seven six zero and then twenty nine twenty nine.", # Mixed format
        "Is John Doe still there?"
    ]
    content = json.load(open("./tran.json"))

    transcript = [turn["text"] if not turn["speaker"].startswith("agent-tool") else "" for turn in content]
    transcript = [turn for turn in transcript if turn != ""]
    print(len(transcript))

    for turn in transcript:
        nlp = spacy.load("en_core_web_trf")
        doc = nlp(turn)
        for ent in doc.ents:
            print(ent.text, ent.label_)
    # redactor = MedicalRedactor()
    # for line in redactor.redact(transcript):
    #     print(line)
