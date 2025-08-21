import json
import dspy
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# ------------------- Verbose signature -------------------
class MedicalSummarySignature(dspy.Signature):
    """DSPy signature for generating structured medical summaries from text chunks with entities and relationships."""
    
    chunk_content = dspy.InputField(desc="The full text content of the medical chunk/paragraph. Example: After heart surgery for infection, the patient admitted alcohol abuse. A stethoscope exam was unremarkable. Cholesterol test increased to 150 mg/dL; blood pressure measured 120/80. Aspirin therapy was initiated.")
    entities_data = dspy.InputField(desc="JSON string containing the entities and relationships extracted from the chunk")
    
    medical_summary = dspy.OutputField(desc="""You are a medical summarizer. Generate a structured summary using ONLY the entities and relationships provided in the entities_data JSON.

IMPORTANT: Base your summary ENTIRELY on the entities and relationships provided. Each entity has:
- entity_name: the medical term/concept
- entity_type: the category (e.g., Drug, Procedure, Disease, Anatomical Structure)
- entity_description: detailed description

Use the entity_type to determine which category each entity belongs to. Use relationships to understand connections between entities.

Respond with ONLY a JSON string in this exact format (fill in relevant values based on entities, leave empty string for missing):
{
  "ANATOMICAL_STRUCTURE": <Mention any anatomical structures specifically discussed>,
  "BODY_FUNCTION": <List any body functions highlighted>,
  "BODY_MEASUREMENT": <Include normal measurements like blood pressure or temperature>,
  "BM_RESULT": <Results of these measurements>,
  "BM_UNIT": <Units for each measurement>,
  "BM_VALUE": <Values of these measurements>,
  "LABORATORY_DATA": <Outline any laboratory tests mentioned>,
  "LAB_RESULT": <Outcomes of these tests (e.g., 'increased', 'decreased')>,
  "LAB_VALUE": <Specific values from the tests>,
  "LAB_UNIT": <Units of measurement for these values>,
  "MEDICINE": <Name medications discussed>,
  "MED_DOSE": <Dose prescribed (e.g., '500 mg')>,
  "MED_DURATION": <Duration of use (e.g., '7 days')>,
  "MED_FORM": <Form of medication (e.g., 'tablet', 'injection')>,
  "MED_FREQUENCY": <How often the medication is taken (e.g., 'twice daily')>,
  "MED_ROUTE": <Route of administration (e.g., 'oral', 'IV')>,
  "MED_STATUS": <Status of the medication (e.g., 'ongoing', 'discontinued')>,
  "MED_STRENGTH": <Strength or concentration (e.g., '5%')>,
  "MED_UNIT": <Unit for medication (e.g., 'mg', 'ml')>,
  "MED_TOTALDOSE": <Total dose over the full course (e.g., '3500 mg')>,
  "PROBLEM": <Identify any medical conditions or findings>,
  "PROCEDURE": <Describe any procedures>,
  "PROCEDURE_RESULT": <Outcomes of these procedures>,
  "PROC_METHOD": <Methods used for procedures (e.g., 'laparoscopic')>,
  "SEVERITY": <Severity of the conditions mentioned>,
  "MEDICAL_DEVICE": <List any medical devices used>,
  "SUBSTANCE_ABUSE": <Note any substance abuse mentioned>
}
Remember to follow the instructions in the angle brackets and remove any angle brackets from the final JSON string response.
Example:
{
    "ANATOMICAL_STRUCTURE": "heart",
    "MEDICINE": "aspirin",
    "PROBLEM": "infection",
    "PROCEDURE": "surgery"
    "MEDICAL_DEVICE": "stethoscope"
    "SUBSTANCE_ABUSE": "alcohol"
    "LABORATORY_DATA": "cholesterol test"
    "LAB_RESULT": "increased"
    "LAB_VALUE": "150"
    "LAB_UNIT": "mg/dL"
    "BM_RESULT": "120/80"
}                                    
""")

# ------------------- Slim fallback signature -------------------
class MedicalSummarySlimSignature(dspy.Signature):
    """A concise fallback prompt used if the verbose prompt returns no data."""

    chunk_content = dspy.InputField(desc="Medical paragraph text")
    entities_flat = dspy.InputField(desc="Comma-separated entity names present in the paragraph")

    medical_summary = dspy.OutputField(desc="Return ONLY JSON object with any of these keys that appear: ANATOMICAL_STRUCTURE, MEDICINE, PROBLEM, PROCEDURE. Use empty strings for missing keys. Example: {\"MEDICINE\": \"aspirin\", \"PROBLEM\": \"infection\"}")


class EntitySummarizer:
    """Generate structured medical summaries from processed chunks using DSPy and LLM."""
    
    def __init__(self, ollama_model: str = "llama3.1:8b", ollama_endpoint: str = "http://localhost:11434", provider: str = "ollama", model_id: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize the Entity Summarizer.
        
        Args:
            ollama_model: Ollama model to use for summarization (default: llama3.1:8b)
            ollama_endpoint: Ollama endpoint URL
            provider: Provider type ("ollama", "hf", "lmstudio")
            model_id: Model ID for HuggingFace or LM Studio
            endpoint: Endpoint URL for LM Studio
        """
        # Provider/model selection
        self.provider = provider
        self.ollama_model = model_id or ollama_model
        self.ollama_endpoint = endpoint or ollama_endpoint
        self.hf_pipeline = None
        
        # Initialize LLM backend for summarization
        try:
            if self.provider == "lmstudio":
                # Initialize LM Studio - no setup needed, just store parameters
                logger.info(f"Using LM Studio endpoint: {self.ollama_endpoint}")
                self.summary_predictor = None
                self.hf_pipeline = None
            elif self.provider == "hf":
                from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
                import torch
                model_id_resolved = self.ollama_model
                logger.info(f"Initializing HF summarizer model: {model_id_resolved}")
                tokenizer = AutoTokenizer.from_pretrained(model_id_resolved, trust_remote_code=True)

                # Capability checks
                has_accelerate = False
                try:
                    import accelerate  # noqa: F401
                    has_accelerate = True
                except Exception:
                    has_accelerate = False

                # Determine 4-bit support without importing bitsandbytes (avoid metadata errors)
                import importlib.util
                supports_bnb_4bit = False  # Force-disable 4-bit to avoid bitsandbytes

                load_kwargs = {"trust_remote_code": True}
                pipeline_kwargs = {"do_sample": True, "temperature": 0.3}

                if torch.cuda.is_available():
                    if has_accelerate:
                        load_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
                        logger.info("[OK] CUDA available; loading FP16 with device_map=auto")
                    else:
                        load_kwargs.update({"torch_dtype": torch.float16})
                        logger.info("[OK] CUDA available; loading FP16 without device_map (install 'accelerate' for sharding)")
                        pipeline_kwargs["device"] = 0
                else:
                    logger.info("ℹ️ CUDA not available; loading on CPU (may be slow)")
                    pipeline_kwargs["device"] = -1

                # Load and sanitize config to ignore any baked-in 4-bit quantization hints
                try:
                    config = AutoConfig.from_pretrained(model_id_resolved, trust_remote_code=True)
                    if hasattr(config, "quantization_config"):
                        setattr(config, "quantization_config", None)
                except Exception:
                    config = None

                model = AutoModelForCausalLM.from_pretrained(
                    model_id_resolved,
                    config=config,
                    **load_kwargs,
                )
                self.hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, **pipeline_kwargs)
                self.summary_predictor = None
                logger.info("[OK] HF summarizer initialized")
            else:
                # Try different DSPy initialization methods for compatibility
                try:
                    # Method 1: Try newer DSPy initialization
                    import dspy
                    ollama_model_string = f"ollama/{self.ollama_model}"
                    self.lm = dspy.OllamaLocal(
                        model=self.ollama_model,
                        base_url=self.ollama_endpoint,
                        temperature=0.5,
                        max_tokens=500
                    )
                    dspy.settings.configure(lm=self.lm)
                    self.summary_predictor = dspy.Predict(MedicalSummarySignature)
                    logger.info(f"[OK] EntitySummarizer initialized with Ollama model: {self.ollama_model}")
                except AttributeError:
                    try:
                        # Method 2: Try alternative DSPy initialization
                        self.lm = dspy.Ollama(
                            model=self.ollama_model,
                            base_url=self.ollama_endpoint,
                            temperature=0.5,
                            max_tokens=500
                        )
                        dspy.settings.configure(lm=self.lm)
                        self.summary_predictor = dspy.Predict(MedicalSummarySignature)
                        logger.info(f"[OK] EntitySummarizer initialized with Ollama model: {self.ollama_model}")
                    except AttributeError:
                        # Method 3: Fallback to basic initialization
                        logger.warning("DSPy Ollama integration not available, disabling summarizer")
                        self.summary_predictor = None
                        self.lm = None
        except Exception as e:
            logger.error(f"Failed to initialize EntitySummarizer: {e}")
            raise RuntimeError("Could not initialize summarizer backend")
    
    def _parse_summary_response(self, raw_response: str) -> Dict[str, str]:
        """
        Parse the LLM response to extract the structured medical summary.
        
        Args:
            raw_response: Raw response from the LLM
            
        Returns:
            Dictionary containing the structured medical categories
        """
        try:
            # Clean the response
            response_text = str(raw_response).strip()
            
            # Remove markdown code blocks if present
            import re
            response_text = re.sub(r"^```(?:json)?|```$", "", response_text, flags=re.IGNORECASE).strip()
            
            # First try: Direct JSON parsing (when LLM returns JSON directly)
            try:
                summary_data = json.loads(response_text)
                if isinstance(summary_data, dict):
                    # Filter out empty values for cleaner output
                    filtered_data = {k: v for k, v in summary_data.items() if v and v.strip()}
                    if filtered_data:  # Only return if we have actual data
                        return filtered_data
                    else:
                        logger.debug("LLM returned empty medical summary, attempting text parsing")
            except json.JSONDecodeError:
                logger.debug("Response is not valid JSON, attempting text parsing")
            
            # Second try: Look for medical_summary field in DSPy format
            if "medical_summary:" in response_text.lower():
                # Extract content after "medical_summary:"
                parts = response_text.split(":", 1)
                if len(parts) > 1:
                    json_part = parts[1].strip()
                    try:
                        summary_data = json.loads(json_part)
                        if isinstance(summary_data, dict):
                            filtered_data = {k: v for k, v in summary_data.items() if v and v.strip()}
                            if filtered_data:
                                return filtered_data
                    except json.JSONDecodeError:
                        pass
            
            # Third try: Extract key-value pairs from text format
            fallback_summary = {}
            medical_categories = [
                "ANATOMICAL_STRUCTURE", "BODY_FUNCTION", "BODY_MEASUREMENT", "BM_RESULT", 
                "BM_UNIT", "BM_VALUE", "LABORATORY_DATA", "LAB_RESULT", "LAB_VALUE", 
                "LAB_UNIT", "MEDICINE", "MED_DOSE", "MED_DURATION", "MED_FORM", 
                "MED_FREQUENCY", "MED_ROUTE", "MED_STATUS", "MED_STRENGTH", "MED_UNIT", 
                "MED_TOTALDOSE", "PROBLEM", "PROCEDURE", "PROCEDURE_RESULT", "PROC_METHOD", 
                "SEVERITY", "MEDICAL_DEVICE", "SUBSTANCE_ABUSE"
            ]
            
            # Try to extract information from text format
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().upper().replace('"', '').replace("'", "")
                        value = parts[1].strip().replace('"', '').replace("'", "").rstrip(',')
                        if key in medical_categories and value and value.lower() not in ['', 'null', 'none']:
                            fallback_summary[key] = value
            
            # Only return fallback if we found some data
            if fallback_summary:
                return fallback_summary
            
            logger.warning("No medical information could be extracted from the response")
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to parse summary response: {e}")
            return {}
    
    def _create_enhanced_summary_prompt(self, chunk_content: str, entities: List[Dict[str, Any]], 
                                       relationships: List[Dict[str, Any]]) -> str:
        """Create an enhanced prompt that emphasizes entity and relationship information."""
        # Create entity summary by type
        entity_by_type = {}
        for ent in entities:
            etype = ent.get('entity_type', 'Unknown')
            if etype not in entity_by_type:
                entity_by_type[etype] = []
            entity_by_type[etype].append({
                'name': ent.get('entity_name', ''),
                'description': ent.get('entity_description', '')
            })
        
        # Format entities by type
        entity_summary = "ENTITIES BY TYPE:\n"
        for etype, ents in entity_by_type.items():
            entity_summary += f"\n{etype}:\n"
            for e in ents:
                entity_summary += f"  - {e['name']}: {e['description']}\n"
        
        # Format relationships
        relationship_summary = "\nRELATIONSHIPS:\n"
        if relationships:
            for rel in relationships:
                relationship_summary += f"  - {rel.get('source_entity', '')} -> {rel.get('target_entity', '')}: {rel.get('relationship_description', '')}\n"
        else:
            relationship_summary += "  None identified\n"
        
        # Combine into enhanced prompt
        enhanced = f"{chunk_content}\n\n{entity_summary}\n{relationship_summary}"
        return enhanced[:3000]  # Limit length to avoid token limits
    
    def _generate_entity_based_summary(self, entities: List[Dict[str, Any]], 
                                      relationships: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate a medical summary directly from entities and relationships."""
        summary = {}
        
        # Process entities by type
        for ent in entities:
            etype = ent.get('entity_type', '').lower()
            name = ent.get('entity_name', '')
            desc = ent.get('entity_description', '')
            
            # Map entity types to medical categories
            if any(term in etype for term in ['drug', 'medication', 'pharmaceutical', 'chemical']):
                existing = summary.get('MEDICINE', '')
                summary['MEDICINE'] = f"{existing}, {name}" if existing else name
                
            elif any(term in etype for term in ['procedure', 'surgery', 'treatment', 'therapeutic']):
                existing = summary.get('PROCEDURE', '')
                summary['PROCEDURE'] = f"{existing}, {name}" if existing else name
                
            elif any(term in etype for term in ['disease', 'disorder', 'syndrome', 'condition', 'problem']):
                existing = summary.get('PROBLEM', '')
                summary['PROBLEM'] = f"{existing}, {name}" if existing else name
                
            elif any(term in etype for term in ['anatomical', 'organ', 'body part', 'tissue']):
                existing = summary.get('ANATOMICAL_STRUCTURE', '')
                summary['ANATOMICAL_STRUCTURE'] = f"{existing}, {name}" if existing else name
                
            elif any(term in etype for term in ['laboratory', 'test', 'diagnostic']):
                existing = summary.get('LABORATORY_DATA', '')
                summary['LABORATORY_DATA'] = f"{existing}, {name}" if existing else name
                
            elif any(term in etype for term in ['device', 'equipment', 'instrument']):
                existing = summary.get('MEDICAL_DEVICE', '')
                summary['MEDICAL_DEVICE'] = f"{existing}, {name}" if existing else name
                
            elif any(term in etype for term in ['substance abuse', 'addiction', 'alcohol', 'drug abuse']):
                existing = summary.get('SUBSTANCE_ABUSE', '')
                summary['SUBSTANCE_ABUSE'] = f"{existing}, {name}" if existing else name
                
            elif any(term in etype for term in ['function', 'process', 'activity']):
                existing = summary.get('BODY_FUNCTION', '')
                summary['BODY_FUNCTION'] = f"{existing}, {name}" if existing else name
        
        # Add relationship context
        if relationships:
            # Look for specific patterns in relationships
            for rel in relationships:
                rel_desc = rel.get('relationship_description', '').lower()
                source = rel.get('source_entity', '')
                target = rel.get('target_entity', '')
                
                # Extract additional context from relationships
                if 'treats' in rel_desc or 'used for' in rel_desc:
                    if 'MEDICINE' not in summary:
                        summary['MEDICINE'] = source
                    if 'PROBLEM' not in summary:
                        summary['PROBLEM'] = target
                        
                elif 'performed' in rel_desc or 'diagnose' in rel_desc:
                    if 'PROCEDURE' not in summary:
                        summary['PROCEDURE'] = source
        
        return summary
    
    def _heuristic_summary_from_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate a simple summary using entity types when the LLM fails."""
        summary: Dict[str, List[str]] = {}
        for ent in entities:
            etype = ent.get('entity_type', '').lower()
            name = ent['entity_name']
            if 'drug' in etype or 'substance' in etype or 'chemical' in etype:
                summary.setdefault('MEDICINE', []).append(name)
            elif 'procedure' in etype or 'surgery' in etype:
                summary.setdefault('PROCEDURE', []).append(name)
            elif 'disease' in etype or 'problem' in etype or 'disorder' in etype:
                summary.setdefault('PROBLEM', []).append(name)
            elif 'anatomical' in etype or 'body' in etype:
                summary.setdefault('ANATOMICAL_STRUCTURE', []).append(name)
        # Convert lists to comma-separated strings
        return {k: ', '.join(v) for k, v in summary.items()}

    def summarize_chunk(self, chunk_content: str, entities: List[Dict[str, Any]], 
                       relationships: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate a structured medical summary for a chunk with its entities and relationships.
        
        Args:
            chunk_content: The text content of the chunk
            entities: List of entities extracted from the chunk
            relationships: List of relationships between entities
            
        Returns:
            Dictionary containing structured medical summary categories
        """
        try:
            # Skip if no content or entities
            if not chunk_content.strip() or not entities:
                logger.debug("Skipping medical summary - no content or entities")
                return {}
            
            # Prepare entities and relationships data for the LLM (keep only essential fields)
            filtered_entities = []
            for entity in entities:
                # Keep only the essential fields for medical summarization
                filtered_entity = {
                    'entity_name': entity.get('entity_name', ''),
                    'entity_type': entity.get('entity_type', ''),
                    'entity_description': entity.get('entity_description', '')
                }
                filtered_entities.append(filtered_entity)
            
            # Filter relationships to keep only essential fields
            filtered_relationships = []
            for rel in relationships:
                filtered_rel = {
                    'source_entity': rel.get('source_entity', ''),
                    'target_entity': rel.get('target_entity', ''),
                    'relationship_description': rel.get('relationship_description', '')
                }
                filtered_relationships.append(filtered_rel)
            
            entities_data = {
                "entities": filtered_entities,
                "relationships": filtered_relationships
            }
            entities_json = json.dumps(entities_data, indent=2)
            
            logger.debug(f"Generating medical summary for chunk (length: {len(chunk_content)} chars, "
                        f"{len(entities)} entities, {len(relationships)} relationships)")
            
            # Generate summary using HF pipeline if configured
            if self.hf_pipeline is not None:
                try:
                    instruction = (
                        "You are a medical summarizer. Extract medical information from the text and entities. "
                        "Return ONLY valid JSON with the exact keys listed. Fill in relevant values, leave empty string for missing info. "
                        "Do not include any text before or after the JSON."
                    )
                    schema = (
                        '{\n'
                        '  "ANATOMICAL_STRUCTURE": "",\n'
                        '  "BODY_FUNCTION": "",\n'
                        '  "BODY_MEASUREMENT": "",\n'
                        '  "BM_RESULT": "",\n'
                        '  "BM_UNIT": "",\n'
                        '  "BM_VALUE": "",\n'
                        '  "LABORATORY_DATA": "",\n'
                        '  "LAB_RESULT": "",\n'
                        '  "LAB_VALUE": "",\n'
                        '  "LAB_UNIT": "",\n'
                        '  "MEDICINE": "",\n'
                        '  "MED_DOSE": "",\n'
                        '  "MED_DURATION": "",\n'
                        '  "MED_FORM": "",\n'
                        '  "MED_FREQUENCY": "",\n'
                        '  "MED_ROUTE": "",\n'
                        '  "MED_STATUS": "",\n'
                        '  "MED_STRENGTH": "",\n'
                        '  "MED_UNIT": "",\n'
                        '  "MED_TOTALDOSE": "",\n'
                        '  "PROBLEM": "",\n'
                        '  "PROCEDURE": "",\n'
                        '  "PROCEDURE_RESULT": "",\n'
                        '  "PROC_METHOD": "",\n'
                        '  "SEVERITY": "",\n'
                        '  "MEDICAL_DEVICE": "",\n'
                        '  "SUBSTANCE_ABUSE": ""\n'
                        '}'
                    )
                    prompt = (
                        f"{instruction}\n\n"
                        f"Source text:\n{chunk_content}\n\n"
                        f"Entities and relationships (JSON):\n{entities_json}\n\n"
                        f"Respond with JSON only and follow this schema (use empty strings for missing fields):\n{schema}"
                    )
                    pad_id = getattr(self.hf_pipeline.tokenizer, 'eos_token_id', None)
                    gen_kwargs = {
                        'max_new_tokens': 600,
                        'return_full_text': False,
                        'do_sample': False,
                        'temperature': 0.0,
                    }
                    if pad_id is not None:
                        gen_kwargs['pad_token_id'] = pad_id
                        gen_kwargs['eos_token_id'] = pad_id
                    outputs = self.hf_pipeline(prompt, **gen_kwargs)
                    raw_response = outputs[0].get('generated_text', '')
                    summary = self._parse_summary_response(raw_response)
                    if not summary:
                        logging.getLogger("entity_processor1").warning(
                            "HF summarizer returned no JSON summary – skipping fallback and logging only"
                        )
                        return {}
                    logger.debug(f"Generated medical summary with {len(summary)} relevant categories (HF)")
                    return summary
                except Exception as hf_error:
                    logger.warning(f"HF summarization failed: {hf_error}")
                    return {}
            
            # Generate summary using LM Studio if configured
            if self.provider == "lmstudio":
                try:
                    import requests
                    
                    instruction = (
                        "You are a medical summarizer. Extract medical information from the text and entities. "
                        "Return ONLY valid JSON with the exact keys listed. Fill in relevant values, leave empty string for missing info. "
                        "Do not include any text before or after the JSON."
                    )
                    schema = (
                        '{\n'
                        '  "ANATOMICAL_STRUCTURE": "",\n'
                        '  "BODY_FUNCTION": "",\n'
                        '  "BODY_MEASUREMENT": "",\n'
                        '  "BM_RESULT": "",\n'
                        '  "BM_UNIT": "",\n'
                        '  "BM_VALUE": "",\n'
                        '  "LABORATORY_DATA": "",\n'
                        '  "LAB_RESULT": "",\n'
                        '  "LAB_VALUE": "",\n'
                        '  "LAB_UNIT": "",\n'
                        '  "MEDICINE": "",\n'
                        '  "MED_DOSE": "",\n'
                        '  "MED_DURATION": "",\n'
                        '  "MED_FORM": "",\n'
                        '  "MED_FREQUENCY": "",\n'
                        '  "MED_ROUTE": "",\n'
                        '  "MED_STATUS": "",\n'
                        '  "MED_STRENGTH": "",\n'
                        '  "MED_UNIT": "",\n'
                        '  "MED_TOTALDOSE": "",\n'
                        '  "PROBLEM": "",\n'
                        '  "PROCEDURE": "",\n'
                        '  "PROCEDURE_RESULT": "",\n'
                        '  "PROC_METHOD": "",\n'
                        '  "SEVERITY": "",\n'
                        '  "MEDICAL_DEVICE": "",\n'
                        '  "SUBSTANCE_ABUSE": ""\n'
                        '}'
                    )
                    prompt = (
                        f"{instruction}\n\n"
                        f"Source text:\n{chunk_content}\n\n"
                        f"Entities and relationships (JSON):\n{entities_json}\n\n"
                        f"Respond with JSON only and follow this schema (use empty strings for missing fields):\n{schema}"
                    )
                    
                    url = f"{self.ollama_endpoint}/v1/chat/completions"
                    payload = {
                        "model": self.ollama_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a medical expert. Always respond with valid JSON only."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        "temperature": 0.0,
                        "max_tokens": 600,
                        "stream": False
                    }
                    
                    resp = requests.post(url, json=payload, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    raw_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    summary = self._parse_summary_response(raw_response)
                    
                    if not summary:
                        logger.warning("LM Studio summarizer returned no JSON summary")
                        return {}
                        
                    logger.debug(f"Generated medical summary with {len(summary)} relevant categories (LM Studio)")
                    return summary
                    
                except Exception as lms_error:
                    logger.warning(f"LM Studio summarization failed: {lms_error}")
                    return {}

            # Generate summary using DSPy (Ollama) when HF and LM Studio are not configured
            if self.summary_predictor is not None:
                try:
                    # Create a more detailed prompt that emphasizes using entity information
                    enhanced_prompt = self._create_enhanced_summary_prompt(chunk_content, filtered_entities, filtered_relationships)
                    
                    result = self.summary_predictor(
                        chunk_content=enhanced_prompt,
                        entities_data=entities_json
                    )
                    
                    # Extract the medical summary JSON from DSPy result
                    raw_response = ""
                    if hasattr(result, 'medical_summary'):
                        raw_response = result.medical_summary
                        logger.debug(f"DSPy medical_summary field: {raw_response[:200]}...")
                    else:
                        # Fallback: try to get raw completion text
                        raw_response = str(result)
                        logger.debug(f"DSPy fallback response: {raw_response[:200]}...")
                    
                    # Parse the response
                    summary = self._parse_summary_response(raw_response)
                    
                    if not summary:
                        # Try a more direct approach with entity-based summary
                        logger.debug("Attempting entity-based summary generation")
                        summary = self._generate_entity_based_summary(filtered_entities, filtered_relationships)
                    
                    if not summary:
                        logging.getLogger("entity_processor1").warning(
                            "LLM medical summary empty – skipping fallback summary and logging only"
                        )
                        return {}
                    else:
                        logger.debug(f"Generated medical summary with {len(summary)} relevant categories")
                    return summary
                    
                except Exception as dspy_error:
                    logger.warning(f"DSPy prediction failed: {dspy_error}")
                    
                    # Try to extract any useful information from the error
                    error_str = str(dspy_error)
                    if "LM Response:" in error_str:
                        # Extract the LM response from the error message
                        response_start = error_str.find("LM Response:") + len("LM Response:")
                        response_end = error_str.find("\n\n", response_start)
                        if response_end == -1:
                            response_end = len(error_str)
                        
                        raw_response = error_str[response_start:response_end].strip()
                        logger.debug(f"Extracted response from error: {raw_response}")
                        
                        # Try to parse this extracted response
                        summary = self._parse_summary_response(raw_response)
                        if summary:
                            logger.info("Successfully recovered medical summary from error message")
                            return summary
                    
                    logger.warning("Could not recover medical summary from DSPy error")
                    return {}
            else:
                # DSPy is not available, use fallback entity-based summary
                logger.debug("DSPy not available, using entity-based summary generation")
                summary = self._generate_entity_based_summary(filtered_entities, filtered_relationships)
                if summary:
                    logger.debug(f"Generated fallback medical summary with {len(summary)} relevant categories")
                    return summary
                else:
                    logger.debug("No entities suitable for fallback summary generation")
                    return {}
            
        except Exception as e:
            logger.warning(f"Failed to generate medical summary for chunk: {e}")
            return {}
    
    def summarize_paragraph(self, paragraph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add medical summary to a paragraph result structure.
        
        Args:
            paragraph_data: Dictionary containing paragraph content, entities, and relationships
            
        Returns:
            Updated paragraph data with medical summary added
        """
        try:
            # Extract data from paragraph structure
            content = paragraph_data.get('content', '')
            entities = paragraph_data.get('entities', [])
            relationships = paragraph_data.get('relationships', [])
            
            # Generate summary
            medical_summary = self.summarize_chunk(content, entities, relationships)
            
            # Add summary to paragraph data (after relationships as requested)
            updated_paragraph = paragraph_data.copy()
            updated_paragraph['medical_summary'] = medical_summary
            
            return updated_paragraph
            
        except Exception as e:
            logger.warning(f"Failed to add medical summary to paragraph: {e}")
            # Return original data if summarization fails
            return paragraph_data


def summarize_entities_for_results(results: Dict[str, Any], 
                                 ollama_model: str = "llama3.1:8b",
                                 ollama_endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Add medical summaries to all paragraphs in entity processing results.
    
    Args:
        results: Entity processing results dictionary
        ollama_model: Ollama model to use for summarization
        ollama_endpoint: Ollama endpoint URL
        
    Returns:
        Updated results with medical summaries added to each paragraph
    """
    try:
        # Initialize summarizer
        summarizer = EntitySummarizer(ollama_model=ollama_model, ollama_endpoint=ollama_endpoint)
        
        # Process each paragraph
        updated_results = results.copy()
        updated_paragraphs = []
        
        paragraphs = results.get('paragraphs', [])
        total_paragraphs = len(paragraphs)
        
        logger.info(f"Adding medical summaries to {total_paragraphs} paragraphs...")
        
        for idx, paragraph in enumerate(paragraphs):
            logger.info(f"Generating medical summary for paragraph {idx + 1}/{total_paragraphs}")
            
            # Add medical summary to paragraph
            updated_paragraph = summarizer.summarize_paragraph(paragraph)
            updated_paragraphs.append(updated_paragraph)
        
        # Update results
        updated_results['paragraphs'] = updated_paragraphs
        
        # Update metadata
        if 'metadata' in updated_results:
            updated_results['metadata']['medical_summaries_added'] = True
            updated_results['metadata']['summarization_model'] = ollama_model
        
        logger.info(f"[OK] Medical summaries added to {total_paragraphs} paragraphs")
        return updated_results
        
    except Exception as e:
        logger.error(f"Failed to add medical summaries to results: {e}")
        return results  # Return original results if summarization fails


# Example usage and integration
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python summarize_entities.py <entity_processing_results.json> [ollama_model]")
        sys.exit(1)
    
    results_path = sys.argv[1]
    ollama_model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b"
    
    try:
        # Load existing entity processing results
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Add medical summaries
        updated_results = summarize_entities_for_results(results, ollama_model=ollama_model)
        
        # Save updated results
        output_path = results_path.replace('.json', '_with_summaries.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_results, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Results with medical summaries saved to {output_path}")
        
        # Print summary statistics
        total_summaries = sum(1 for p in updated_results.get('paragraphs', []) 
                             if p.get('medical_summary'))
        print(f"Added medical summaries to {total_summaries} paragraphs")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)