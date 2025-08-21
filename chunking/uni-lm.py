#!/usr/bin/env python3
"""
Unified PDF Processing Pipeline - Steps 1 & 2 Only
==================================================
This script integrates PDF to Markdown conversion and semantic chunking.
Step 3 (LLM entity processing) has been replaced with a placeholder.

Usage:
    python uni-lm.py <path_to_pdf> --output-dir <output_directory>
"""

import os
import sys
import json
import argparse
import time
import traceback
import textwrap
import re
import csv
import html
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import hashlib
from collections import defaultdict

# Add better error handling for imports
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError as exc:
    print(f"ERROR: Missing required Marker library: {exc}")
    print("Please install with: pip install marker-pdf")
    MARKER_AVAILABLE = False

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError as exc:
    print(f"Warning: {exc}. Ollama integration will not be available.")
    dspy = None
    DSPY_AVAILABLE = False

from tqdm import tqdm


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def discover_files(input_path: str) -> List[str]:
    """Discover all supported files in the given path.
    
    Args:
        input_path: Can be a file path or directory path
        
    Returns:
        List of file paths to process
    """
    supported_extensions = {'.pdf', '.md', '.markdown'}
    files_to_process = []
    
    if os.path.isfile(input_path):
        # Single file - validate extension
        ext = os.path.splitext(input_path)[1].lower()
        if ext in supported_extensions:
            files_to_process.append(input_path)
        else:
            print(f"Warning: Unsupported file type '{ext}' for file: {input_path}")
    elif os.path.isdir(input_path):
        # Directory - find all supported files
        print(f"Scanning directory: {input_path}")
        for root, dirs, files in os.walk(input_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    full_path = os.path.join(root, file)
                    files_to_process.append(full_path)
        
        if not files_to_process:
            print(f"No supported files found in directory: {input_path}")
            print(f"Supported extensions: {', '.join(supported_extensions)}")
    else:
        print(f"Error: Path does not exist: {input_path}")
    
    return sorted(files_to_process)


# =============================================================================
# SECTION 1: PDF CONVERSION (Same as unified.py)
# =============================================================================

class MarkerPDFConverter:
    """Handles the conversion of PDF files to Markdown format using Marker."""

    class MarkerJSONEncoder(json.JSONEncoder):
        """Custom JSON encoder that handles non-serializable objects from Marker."""
        def default(self, obj):
            if hasattr(obj, 'mode') and hasattr(obj, 'size'):
                return f"<Image: {obj.size[0]}x{obj.size[1]} {getattr(obj, 'mode', 'unknown')}>"
            if hasattr(obj, '__dict__'):
                try:
                    return obj.__dict__
                except:
                    return f"<{type(obj).__name__}: non-serializable>"
            if isinstance(obj, set):
                return list(obj)
            try:
                return str(obj)
            except:
                return f"<{type(obj).__name__}: non-serializable>"

    def clean_data_for_json(self, data):
        """Recursively clean data to make it JSON serializable."""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                try:
                    json.dumps(value, cls=self.MarkerJSONEncoder)
                    cleaned[key] = self.clean_data_for_json(value)
                except (TypeError, ValueError):
                    cleaned[key] = f"<{type(value).__name__}: non-serializable>"
            return cleaned
        elif isinstance(data, list):
            return [self.clean_data_for_json(item) for item in data]
        elif isinstance(data, set):
            return list(data)
        else:
            return data

    def convert(self, pdf_path: str) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        """Convert PDF using Marker."""
        print("  • Converting PDF with Marker...")
        try:
            converter = PdfConverter(artifact_dict=create_model_dict())
            rendered = converter(pdf_path)
            text, _, _ = text_from_rendered(rendered)
            markdown_text = text
            
            try:
                if hasattr(rendered, 'model_dump'):
                    raw_json_data = rendered.model_dump()
                elif hasattr(rendered, 'dict'):
                    raw_json_data = rendered.dict()
                else:
                    raw_json_data = {"text": text, "type": "conversion_output", "source": "marker"}
                json_data = self.clean_data_for_json(raw_json_data)
            except Exception as e:
                print(f"  • Warning: Could not extract full JSON data: {e}")
                json_data = {"text": text, "type": "conversion_output", "source": "marker", "error": str(e)}

            try:
                metadata = getattr(rendered, 'metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {"source": "marker", "status": "converted"}
                metadata = self.clean_data_for_json(metadata)
            except Exception:
                metadata = {"source": "marker", "status": "converted"}
            
            return markdown_text, text, json_data, metadata
        except Exception as e:
            print(f"  • Error in Marker conversion: {e}")
            raise

    def apply_text_reconstruction(self, raw_text: str, model: str, endpoint: str) -> str:
        """Use Ollama to reconstruct logical reading order and fix sentence breaks."""
        if not DSPY_AVAILABLE:
            print("  • Warning: DSPy not available, skipping Ollama reconstruction")
            return raw_text
        if not raw_text.strip():
            return raw_text
        
        print(f"  • Applying Ollama text reconstruction with model: {model}")
        api_base = endpoint.split("/api", 1)[0] if "/api" in endpoint else endpoint.rstrip("/")
        
        try:
            full_model_name = f"ollama/{model}"
            print(f"  • Using full model name: {full_model_name}")
            lm = dspy.LM(
                model=full_model_name,
                temperature=0.0,
                max_tokens=4096,
                api_base=api_base,
            )
            
            system_prompt = (
                "You are an expert text reconstruction specialist. The user will provide raw text that "
                "may have reading order issues from a multi-column PDF or document. "
                "Your task is to: "
                "1) Reconstruct the logical reading order "
                "2) Fix obvious sentence breaks and formatting issues "
                "3) Preserve all original content without adding or removing information "
                "4) Maintain proper paragraph structure "
                "5) Keep tables, lists, and special formatting intact "
                "Return ONLY the reconstructed text without any additional commentary."
            )
            user_prompt = f"Please reconstruct the following text by fixing reading order and sentence breaks:\n\n{raw_text}"
            
            # Combine system and user prompts for simple string input
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = lm(full_prompt)
            reconstructed_text = response[0] if isinstance(response, list) else str(response)
            return reconstructed_text.strip()
        except Exception as e:
            print(f"  • Warning: Ollama text reconstruction failed: {e}")
            print("  • Returning original text")
            return raw_text


# =============================================================================
# SECTION 2: MARKDOWN CHUNKING (Same as unified.py)
# =============================================================================

class MarkdownChunker:
    """Handles the chunking of markdown files into semantic groups with optional content cleaning."""
    
    def __init__(self):
        """Initialize the MarkdownChunker with content cleaning capabilities."""
        self.content_cache = {}
        self.cleaning_stats = defaultdict(int)

    def is_ignored_figure(self, line: str) -> bool:
        if line.startswith('![') and '](' in line:
            url_start = line.find('](') + 2
            url_end = line.find(')', url_start)
            if url_end != -1:
                url = line[url_start:url_end]
                if re.match(r'^_page_\d+_Figure_\d+\.(jpeg|jpg|png|gif|svg)$', url):
                    return True
        return False

    def read_markdown_paragraphs_advanced(self, markdown_file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(markdown_file_path):
          raise FileNotFoundError(f"Markdown file not found: {markdown_file_path}")
        try:
            with open(markdown_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            raise Exception(f"Error reading markdown file: {e}")
        
        lines = content.split('\n')
        paragraphs = []
        current_paragraph = []
        paragraph_type = "text"
        paragraph_level = 0
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line:
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph).strip()
                    if paragraph_text:
                        paragraphs.append({'md-text': paragraph_text, 'type': paragraph_type, 'level': paragraph_level})
                    current_paragraph = []
                    paragraph_type = "text"
                paragraph_level = 0
                continue
            
            if self.is_ignored_figure(stripped_line):
                continue
                
            current_paragraph.append(stripped_line)

        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph).strip()
            if paragraph_text:
                paragraphs.append({'md-text': paragraph_text, 'type': paragraph_type, 'level': paragraph_level})
        
        # Add type detection
        for para in paragraphs:
            para['type'] = self.detect_paragraph_type_from_content(para)
            if para['type'] == 'header':
                 para['level'] = len(para['md-text']) - len(para['md-text'].lstrip('#'))
        
        return paragraphs

    def detect_paragraph_type_from_content(self, paragraph: Dict[str, Any]) -> str:
        content = paragraph.get('md-text', '').strip()
        if not content: return 'empty'
        if content.startswith('#'): return 'header'
        if content.startswith('|') and '|' in content[1:]: return 'table'
        if content.startswith('![') and '](' in content:
            if self.is_ignored_figure(content): return 'ignored_figure'
            return 'image'
        if content.startswith('```'): return 'code'
        if content.startswith('>'): return 'quote'
        if (content.startswith('- ') or content.startswith('* ') or content.startswith('+ ') or re.match(r'^\d+\.\s', content)): return 'list'
        if content in ['---', '***', '___']: return 'rule'
        if content.startswith('[') and '](' in content: return 'link'
        if re.match(r'^\s*\$\$.*\$\$\s*$', content): return 'latex'
        return 'text'

    def create_sliding_windows(self, paragraphs: List[Dict[str, Any]], window_size: int = 5, step_size: int = 1) -> List[List[Dict[str, Any]]]:
        if not paragraphs or window_size <= 0 or step_size <= 0:
            return []
        if window_size > len(paragraphs):
            return [paragraphs]
        
        windows = []
        for i in range(0, len(paragraphs) - window_size + 1, step_size):
            windows.append(paragraphs[i:i + window_size])
        return windows

    def get_content_hash(self, content: str) -> str:
        """Generate a hash for content to enable caching."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def is_obviously_unnecessary(self, content: str, content_type: str) -> Optional[bool]:
        """Quick rule-based check for obviously unnecessary content.
        
        Returns:
            True if definitely unnecessary, False if definitely necessary, None if needs LLM
        """
        content_lower = content.lower().strip()
        
        unnecessary_patterns = [
            r'^the authors? (thank|acknowledge|gratefully)',
            r'^acknowledgements?:?\s*$',
            r'^references?:?\s*$',
            r'^bibliography:?\s*$',
            r'^funding:?\s*',
            r'^conflicts? of interest:?\s*',
            r'^author contributions?:?\s*',
            r'^supplementary materials?:?\s*',
            r'^appendix [a-z]?:?\s*',
            r'^figure \d+[:.]\s*',
            r'^table \d+[:.]\s*',
            r'^\[?\d+\]?\s+[A-Z][a-z]+.*\d{4}',
            r'^copyright\s*©?\s*\d{4}',
            r'^doi:\s*10\.\d+',
            r'^isbn:?\s*[\d\-x]+',
            r'^edited by\s+',
            r'^co-author.*edition',
            r'^dr\.\s+\w+.*professor',
            r'^distinguished professor',
            r'contributions? and mentorship',
        ]
        
        for pattern in unnecessary_patterns:
            if re.match(pattern, content_lower):
                return True
        
        necessary_keywords = [
            'patient', 'treatment', 'diagnosis', 'symptom', 'disease',
            'procedure', 'medication', 'dosage', 'clinical', 'study',
            'result', 'conclusion', 'method', 'analysis', 'data'
        ]
        
        if content_type == 'header' and any(keyword in content_lower for keyword in necessary_keywords):
            return False
        
        if len(content.strip()) < 20 and content_type == 'text':
            if not any(char.isalnum() for char in content):
                return True
        
        return None
    
    def classify_content_necessity(self, content: str, content_type: str, classifier=None) -> Tuple[bool, str]:
        """Classify whether content is necessary using rules and LLM.
        
        Returns:
            Tuple of (is_necessary, reason)
        """
        content_hash = self.get_content_hash(content)
        
        if content_hash in self.content_cache:
            self.cleaning_stats['cache_hits'] += 1
            return self.content_cache[content_hash]
        
        quick_check = self.is_obviously_unnecessary(content, content_type)
        if quick_check is not None:
            self.cleaning_stats['rule_based'] += 1
            reason = "Rule-based classification"
            result = (not quick_check, reason)
            self.content_cache[content_hash] = result
            return result
        
        if classifier is not None:
            try:
                response = classifier(
                    content=content[:500],
                    content_type=content_type
                )
                
                is_necessary = response.necessary.strip().upper() == 'TRUE'
                reason = response.reason.strip()
                
                self.cleaning_stats['llm_classified'] += 1
                result = (is_necessary, reason)
                self.content_cache[content_hash] = result
                return result
            except Exception as e:
                self.cleaning_stats['classification_errors'] += 1
        
        # Default to necessary if no classifier or classification failed
        return (True, "No classification available")
    
    def save_checkpoint(self, data: Dict[str, Any], output_path: str, checkpoint_num: int):
        """Save a checkpoint of the chunking and cleaning progress.
        
        Args:
            data: Data structure with chunks
            output_path: Final output path
            checkpoint_num: Checkpoint number
        """
        checkpoint_path = output_path.replace('.json', f'_checkpoint_{checkpoint_num}.json')
        
        checkpoint_data = data.copy()
        if 'metadata' not in checkpoint_data:
            checkpoint_data['metadata'] = {}
        
        checkpoint_data['metadata']['checkpoint'] = {
            'number': checkpoint_num,
            'chunks_processed': len(checkpoint_data.get('chunks', [])),
            'saved_at': datetime.now().isoformat(),
            'incomplete': True
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Checkpoint {checkpoint_num} saved ({len(checkpoint_data.get('chunks', []))} chunks)")

    def llm_for_context_chunking(self, windows: List[List[Dict[str, Any]]], output_filename: str = None, 
                                  model: str = "llama3.2", endpoint: str = "http://localhost:11434", 
                                  clean_content: bool = False, checkpoint_interval: int = 1000) -> List[Dict[str, Any]]:
        """Use LLM to analyze windows and determine chunk boundaries."""
        if not DSPY_AVAILABLE:
            print("DSPy not available, skipping LLM chunking.")
            # Basic chunking: every 5 paragraphs is a chunk
            chunk_num = 1
            all_paragraphs = []
            processed_texts = set()
            for window in windows:
                for para in window:
                    if para['md-text'] not in processed_texts:
                        all_paragraphs.append(para)
                        processed_texts.add(para['md-text'])
            
            for i, para in enumerate(all_paragraphs):
                if i > 0 and i % 5 == 0:
                    chunk_num += 1
                para['chunk_number'] = chunk_num
                para['chunk_title'] = f"Chunk {chunk_num}"
                para['chunk_description'] = "Basic chunking"
            
            if output_filename:
                self.convert_chunks_to_json(all_paragraphs, output_filename)

            return all_paragraphs

        api_base = endpoint.split("/api", 1)[0] if "/api" in endpoint else endpoint.rstrip("/")
        llm = dspy.LM(model=f"ollama/{model}", api_base=api_base, temperature=0.0, max_tokens=150)
        dspy.settings.configure(lm=llm)
    
        class TopicAnalyzer(dspy.Signature):
            context = dspy.InputField(desc="Previous paragraphs for context")
            current_text = dspy.InputField(desc="Current paragraph to analyze")
            current_chunk = dspy.InputField(desc="Current chunk number")
            decision = dspy.OutputField(desc="Answer 'NEW' to start new chunk or 'CONTINUE' to stay in current chunk")
            reason = dspy.OutputField(desc="Brief reason for decision")
    
        class ChunkTitleGenerator(dspy.Signature):
            chunk_content = dspy.InputField(desc="Content of the new chunk")
            title = dspy.OutputField(desc="Short, descriptive title for the chunk (max 3 words).")
            description = dspy.OutputField(desc="Brief description of what this chunk covers (max 10 words)")
        
        class ContentNecessityClassifier(dspy.Signature):
            """Classify whether content is necessary for understanding the main text."""
            content = dspy.InputField(desc="The text content to analyze")
            content_type = dspy.InputField(desc="The type of content (text, header, table, etc.)")
            necessary = dspy.OutputField(desc="Answer TRUE if content is necessary for understanding main text, FALSE if it's metadata/academic titles/professional titles or descriptions/acknowledgements/references")
            reason = dspy.OutputField(desc="Brief reason for classification (max 10 words)")
    
        analyzer = dspy.Predict(TopicAnalyzer)
        title_generator = dspy.Predict(ChunkTitleGenerator)
        content_classifier = dspy.Predict(ContentNecessityClassifier) if clean_content else None

        enhanced_paragraphs = []
        current_chunk = 1
        context_paragraphs = []
        processed_texts = set()
        current_chunk_title = "Introduction"
        current_chunk_description = "Initial content"

        all_paragraphs = []
        for window in windows:
            for para in window:
                if para['md-text'] not in processed_texts:
                    all_paragraphs.append(para)
                    processed_texts.add(para['md-text'])
        
        processed_texts.clear()

        checkpoint_num = 0
        chunks_processed = 0
        
        desc = "Semantic Chunking" + (" & Cleaning" if clean_content else "")
        for i, para in enumerate(tqdm(all_paragraphs, desc=desc), 1):
            para_text = para.get('md-text', '')
            if para_text in processed_texts:
                continue
            
            should_create_new = False
            reasoning = ""
            if para.get('type') == 'header':
                should_create_new = True
                reasoning = "Header starts a new section."

            if not should_create_new and context_paragraphs:
                context_str = "\n".join([p['md-text'] for p in context_paragraphs[-3:]])
                try:
                    result = analyzer(context=context_str, current_text=para_text, current_chunk=str(current_chunk))
                    if result.decision.strip().upper() == 'NEW':
                        should_create_new = True
                        reasoning = result.reason
                except Exception as e:
                    print(f"LLM topic analysis failed: {e}. Defaulting to CONTINUE.")

            if should_create_new and enhanced_paragraphs:
                current_chunk += 1
                chunks_processed += 1
                
                # Save checkpoint every N chunks
                if checkpoint_interval > 0 and chunks_processed % checkpoint_interval == 0 and output_filename:
                    checkpoint_num += 1
                    # Build temporary document structure
                    temp_doc = {
                        "metadata": {
                            "created_at": datetime.now().isoformat(),
                            "total_paragraphs": len(enhanced_paragraphs),
                            "total_chunks": current_chunk,
                        },
                        "chunks": []
                    }
                    # Convert paragraphs to chunks format for checkpoint
                    chunks_map = defaultdict(lambda: {"paragraphs": []})
                    for ep in enhanced_paragraphs:
                        chunk_num = ep.get('chunk_number', 0)
                        if "title" not in chunks_map[chunk_num]:
                            chunks_map[chunk_num]['chunk_number'] = chunk_num
                            chunks_map[chunk_num]['title'] = ep.get('chunk_title', 'Untitled')
                            chunks_map[chunk_num]['description'] = ep.get('chunk_description', 'No description')
                        
                        para_obj = {
                            "content": ep.get('md-text', ''),
                            "type": ep.get('type', 'text'),
                            "level": ep.get('level', 0)
                        }
                        if clean_content and 'necessary' in ep:
                            para_obj['necessary'] = ep['necessary']
                        chunks_map[chunk_num]["paragraphs"].append(para_obj)
                    
                    temp_doc["chunks"] = list(chunks_map.values())
                    self.save_checkpoint(temp_doc, output_filename, checkpoint_num)
                
                chunk_content_for_title = " ".join([p['md-text'] for p in context_paragraphs])
                try:
                    title_res = title_generator(chunk_content=chunk_content_for_title[:1000])
                    current_chunk_title = title_res.title
                    current_chunk_description = title_res.description
                except Exception as e:
                    print(f"LLM title generation failed: {e}")
                    current_chunk_title = f"Chunk {current_chunk}"
                    current_chunk_description = "Description unavailable"

            para_enhanced = para.copy()
            para_enhanced['chunk_number'] = current_chunk
            para_enhanced['chunk_title'] = current_chunk_title
            para_enhanced['chunk_description'] = current_chunk_description
            
            # Add content cleaning if enabled
            if clean_content and content_classifier:
                para_type = para.get('type', 'text')
                is_necessary, reason = self.classify_content_necessity(para_text, para_type, content_classifier)
                para_enhanced['necessary'] = is_necessary
                if not is_necessary:
                    self.cleaning_stats['unnecessary_paragraphs'] += 1
                else:
                    self.cleaning_stats['necessary_paragraphs'] += 1
            
            enhanced_paragraphs.append(para_enhanced)
            context_paragraphs.append(para_enhanced)
            processed_texts.add(para_text)
            
            if len(context_paragraphs) > 5:
                context_paragraphs.pop(0)
            
        if output_filename:
            self.convert_chunks_to_json(enhanced_paragraphs, output_filename, clean_content, checkpoint_interval)
        
        return enhanced_paragraphs

    def convert_chunks_to_json(self, paragraphs: List[Dict[str, Any]], output_file: str, 
                               clean_content: bool = False, checkpoint_interval: int = 1000):
        document = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_paragraphs": len(paragraphs),
                "total_chunks": len(set(p.get('chunk_number', 0) for p in paragraphs)),
            },
            "chunks": []
        }
        
        # Add cleaning statistics if cleaning was performed
        if clean_content and self.cleaning_stats:
            document["metadata"]["cleaning"] = {
                "unnecessary_paragraphs": self.cleaning_stats['unnecessary_paragraphs'],
                "necessary_paragraphs": self.cleaning_stats['necessary_paragraphs'],
                "cache_hits": self.cleaning_stats['cache_hits'],
                "rule_based": self.cleaning_stats['rule_based'],
                "llm_classified": self.cleaning_stats['llm_classified'],
                "errors": self.cleaning_stats['classification_errors']
            }
        
        chunks_map = defaultdict(lambda: {"paragraphs": []})
        for para in paragraphs:
            chunk_num = para.get('chunk_number', 0)
            if "title" not in chunks_map[chunk_num]:
                 chunks_map[chunk_num]['chunk_number'] = chunk_num
                 chunks_map[chunk_num]['title'] = para.get('chunk_title', 'Untitled')
                 chunks_map[chunk_num]['description'] = para.get('chunk_description', 'No description')
            
            para_obj = {
                "content": para.get('md-text', ''),
                "type": para.get('type', 'text'),
                "level": para.get('level', 0)
            }
            # Preserve necessary field if content was cleaned
            if 'necessary' in para:
                para_obj['necessary'] = para['necessary']
            
            chunks_map[chunk_num]["paragraphs"].append(para_obj)
        
        document["chunks"] = list(chunks_map.values())

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, ensure_ascii=False)
        
        # Clean up checkpoint files after successful save
        if checkpoint_interval > 0:
            checkpoint_num = 1
            while True:
                checkpoint_path = output_file.replace('.json', f'_checkpoint_{checkpoint_num}.json')
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    checkpoint_num += 1
                else:
                    break
            if checkpoint_num > 1:
                print(f"  🧹 Cleaned up {checkpoint_num - 1} checkpoint file(s)")
        
        print(f"Chunked JSON saved to: {output_file}")
        
        if clean_content and self.cleaning_stats:
            print(f"  Content Cleaning Statistics:")
            print(f"    - Necessary paragraphs: {self.cleaning_stats['necessary_paragraphs']}")
            print(f"    - Unnecessary paragraphs: {self.cleaning_stats['unnecessary_paragraphs']}")
            print(f"    - Cache hits: {self.cleaning_stats['cache_hits']}")
            print(f"    - Rule-based classifications: {self.cleaning_stats['rule_based']}")
            print(f"    - LLM classifications: {self.cleaning_stats['llm_classified']}")


# =============================================================================
# SECTION 3: PLACEHOLDER FOR ENTITY PROCESSING
# =============================================================================

class EntityProcessorPlaceholder:
    """Placeholder class for entity processing functionality.
    
    This class replaces the original LLMEntityProcessor and provides
    a simple interface for future entity processing implementations.
    """
    
    def __init__(self):
        print("  • EntityProcessorPlaceholder initialized")
        print("  • This is a placeholder for future entity processing functionality")
    
    def process_entities(self, input_file_path: str, output_file_path: str):
        """Placeholder method for entity processing."""
        print("  • Entity processing placeholder called")
        print(f"  • Input file: {input_file_path}")
        print(f"  • Output file: {output_file_path}")
        print("  • No entity processing performed - this is a placeholder")
        
        # Simply copy the input file to output file as-is
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add placeholder metadata
            data["metadata"]["entity_processing"] = {
                "status": "placeholder",
                "message": "Entity processing not implemented - placeholder only",
                "timestamp": datetime.now().isoformat()
            }
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"  • Placeholder processing completed - file copied to: {output_file_path}")
            
        except Exception as e:
            print(f"  • Error in placeholder processing: {e}")
            raise


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_pipeline_single_file(input_file: str, args):
    """Run the PDF processing pipeline for a single file."""
    
    # Detect file type
    file_extension = os.path.splitext(input_file)[1].lower()
    
    if not MARKER_AVAILABLE and file_extension == '.pdf':
        print(f"Error: Marker library not available for file: {input_file}. Please install with: pip install marker-pdf")
        return False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames (with subdirectory structure to avoid conflicts)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create safe filename by removing problematic characters
    safe_base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
    
    markdown_file = os.path.join(args.output_dir, f"{safe_base_name}.md")
    chunked_json_file = os.path.join(args.output_dir, f"{safe_base_name}_chunked.json")
    final_output_file = os.path.join(args.output_dir, f"{safe_base_name}_processed.json")
    
    try:
        print("\n" + "=" * 80)
        print(f"PROCESSING FILE: {os.path.basename(input_file)}")
        print("=" * 80)
        
        # STEP 1: File to Markdown conversion (or skip if already markdown)
        print(f"\nSTEP 1: {file_extension.upper()} to Markdown Conversion")
        print("-" * 40)
        
        if file_extension == '.pdf':
            # Convert PDF to markdown
            converter = MarkerPDFConverter()
            markdown_text, raw_text, json_data, metadata = converter.convert(input_file)
            
            # Apply text reconstruction if requested
            if args.reconstruct and DSPY_AVAILABLE:
                print("  • Applying text reconstruction...")
                markdown_text = converter.apply_text_reconstruction(
                    markdown_text, 
                    args.ollama_model, 
                    args.ollama_endpoint
                )
            
            # Save markdown file
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            print(f"  • Markdown saved to: {markdown_file}")
            
        else:  # .md or .markdown file
            # Already a markdown file, just copy it to output directory
            print(f"  • Input is already a markdown file: {input_file}")
            print(f"  • Copying to output directory...")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            
            # Apply text reconstruction if requested
            if args.reconstruct and DSPY_AVAILABLE:
                print("  • Applying text reconstruction...")
                converter = MarkerPDFConverter()
                markdown_text = converter.apply_text_reconstruction(
                    markdown_text, 
                    args.ollama_model, 
                    args.ollama_endpoint
                )
            
            # Save markdown file
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            print(f"  • Markdown saved to: {markdown_file}")
        
        # STEP 2: Markdown chunking
        print("\nSTEP 2: Markdown Chunking")
        print("-" * 40)
        
        chunker = MarkdownChunker()
        paragraphs = chunker.read_markdown_paragraphs_advanced(markdown_file)
        print(f"  • Read {len(paragraphs)} paragraphs from markdown")
        
        # Create sliding windows for semantic chunking
        windows = chunker.create_sliding_windows(paragraphs, window_size=5, step_size=1)
        print(f"  • Created {len(windows)} sliding windows")
        
        # Apply LLM-based semantic chunking with optional content cleaning
        enhanced_paragraphs = chunker.llm_for_context_chunking(
            windows, chunked_json_file, args.ollama_model, args.ollama_endpoint,
            clean_content=args.clean, checkpoint_interval=args.checkpoint_interval
        )
        if args.clean:
            print(f"  • Semantic chunking and content cleaning completed - {len(enhanced_paragraphs)} enhanced paragraphs")
        else:
            print(f"  • Semantic chunking completed - {len(enhanced_paragraphs)} enhanced paragraphs")
        
        # STEP 3: Placeholder for entity processing
        print("\nSTEP 3: Entity Processing (Placeholder)")
        print("-" * 40)
        
        entity_processor = EntityProcessorPlaceholder()
        entity_processor.process_entities(chunked_json_file, final_output_file)
        
        print("\n" + "=" * 80)
        print(f"COMPLETED: {os.path.basename(input_file)}")
        print("=" * 80)
        print(f"Output files:")
        print(f"  • Markdown: {markdown_file}")
        print(f"  • Chunked JSON: {chunked_json_file}")
        print(f"  • Final output: {final_output_file}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\nError processing {input_file}: {e}")
        traceback.print_exc()
        return False


def run_pipeline(args):
    """Run the PDF processing pipeline with steps 1 and 2 only."""
    
    # Validate input path
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found.")
        return False
    
    # Discover files to process
    files_to_process = discover_files(args.input)
    
    if not files_to_process:
        print("No files to process.")
        return False
    
    # Check for PDF files when Marker is not available
    pdf_files = [f for f in files_to_process if f.lower().endswith('.pdf')]
    if pdf_files and not MARKER_AVAILABLE:
        print("Error: Marker library not available for PDF processing. Please install with: pip install marker-pdf")
        return False
    
    print(f"\nFound {len(files_to_process)} file(s) to process:")
    for i, file_path in enumerate(files_to_process, 1):
        print(f"  {i:3d}. {file_path}")
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files with progress tracking
    successful_files = []
    failed_files = []
    
    print(f"\nStarting batch processing...")
    print("=" * 80)
    
    for i, input_file in enumerate(files_to_process, 1):
        try:
            print(f"\nProcessing file {i}/{len(files_to_process)}: {os.path.basename(input_file)}")
            
            success = run_pipeline_single_file(input_file, args)
            
            if success:
                successful_files.append(input_file)
                print(f"✓ Successfully processed: {os.path.basename(input_file)}")
            else:
                failed_files.append(input_file)
                print(f"✗ Failed to process: {os.path.basename(input_file)}")
                
        except Exception as e:
            failed_files.append(input_file)
            print(f"✗ Error processing {os.path.basename(input_file)}: {e}")
            if args.debug:
                traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(files_to_process)}")
    print(f"Successful: {len(successful_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if successful_files:
        print("\nSuccessfully processed:")
        for file_path in successful_files:
            print(f"  ✓ {file_path}")
    
    if failed_files:
        print("\nFailed to process:")
        for file_path in failed_files:
            print(f"  ✗ {file_path}")
    
    print(f"\nOutput directory: {args.output_dir}")
    print("=" * 80)
    
    # Return True if at least one file was processed successfully
    return len(successful_files) > 0


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="PDF/Markdown Processing Pipeline - Steps 1 & 2 Only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python uni-lm.py document.pdf --output-dir ./output
  python uni-lm.py document.md --output-dir ./output
  
  # Process all files in a directory
  python uni-lm.py ./documents/ --output-dir ./output
  python uni-lm.py /path/to/pdfs/ --output-dir ./output
  
  # With text reconstruction
  python uni-lm.py ./documents/ --output-dir ./output --reconstruct
  python uni-lm.py document.pdf --output-dir ./output --ollama-model llama3.2
  
  # With content cleaning (single-pass)
  python uni-lm.py document.pdf --output-dir ./output --clean
  python uni-lm.py ./documents/ --output-dir ./output --clean --checkpoint-interval 500
        """
    )
    
    parser.add_argument("input", help="Path to input file or directory. For files: .pdf, .md, .markdown. For directories: processes all supported files recursively.")
    parser.add_argument("--output-dir", "-o", default="./output", 
                       help="Output directory (default: ./output)")
    parser.add_argument("--reconstruct", action="store_true",
                       help="Apply Ollama text reconstruction")
    parser.add_argument("--ollama-model", default="llama3.1:8b",
                       help="Ollama model for text reconstruction (default: llama3.2)")
    parser.add_argument("--ollama-endpoint", default="http://localhost:11434",
                       help="Ollama endpoint (default: http://localhost:11434)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed error traces")
    parser.add_argument("--clean", action="store_true",
                       help="Enable content cleaning to identify unnecessary text (acknowledgements, references, etc.)")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                       help="Save checkpoint every N chunks (default: 1000, 0 to disable)")
    
    args = parser.parse_args()
    
    success = run_pipeline(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 