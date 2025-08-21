#!/usr/bin/env python3
"""
MIMIC Discharge Notes Processing Pipeline
========================================
This script processes discharge notes from the MIMIC dataset and converts them 
into the same chunked JSON format as the unified PDF processing pipeline.

It processes the text column from discharge.csv.gz for the first 10 unique patients
and outputs in the same format as UNMC PDF Transpalnt_chunked.json.

Usage:
    python process_mimic.py [--output-file output.json]
"""

import os
import sys
import json
import argparse
import re
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Add better error handling for imports
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError as exc:
    print(f"Warning: {exc}. Ollama integration will not be available.")
    dspy = None
    DSPY_AVAILABLE = False

class MimicTextProcessor:
    """Processes MIMIC discharge text data into semantic chunks."""

    def clean_text(self, text: str) -> str:
        """Clean the text by handling escape characters and deidentified data."""
        if not text or pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Handle common escape sequences
        text = text.replace('\\n', '\n')
        text = text.replace('\\t', '\t')
        text = text.replace('\\r', '\r')
        text = text.replace('\\\\', '\\')  # Handle double backslashes
        
        # Remove excessive whitespace but preserve paragraph breaks
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Strip whitespace but keep empty lines for paragraph breaks
            cleaned_line = line.strip()
            cleaned_lines.append(cleaned_line)
        
        # Join lines back together
        text = '\n'.join(cleaned_lines)
        
        # Remove multiple consecutive empty lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()

    def remove_escape_characters_final(self, text: str) -> str:
        """Remove all escape characters from final output."""
        if not text:
            return text
        
        # Remove common escape sequences for final output
        escape_chars = {
            '\\n': ' ',
            '\\t': ' ',
            '\\r': ' ',
            '\\\\': '',
            '\\"': '"',
            "\\'": "'",
        }
        
        for escape_seq, replacement in escape_chars.items():
            text = text.replace(escape_seq, replacement)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def split_into_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Split text into paragraphs using both \n and \n\n."""
        if not text:
            return []
        
        paragraphs = []
        
        # First split by double newlines to get major paragraph breaks
        major_sections = re.split(r'\n\s*\n+', text)
        
        for section in major_sections:
            section = section.strip()
            if not section:
                continue
                
            # Then split by single newlines within each section
            lines = section.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect paragraph type
                para_type = self.detect_paragraph_type(line)
                
                paragraph = {
                    'content': line,
                    'type': para_type,
                    'level': 0  # Default level
                }
                
                # Set level for headers
                if para_type == 'header':
                    # Count leading characters that might indicate header level
                    if line.startswith('#'):
                        paragraph['level'] = len(line) - len(line.lstrip('#'))
                    else:
                        # For non-markdown headers, use a heuristic
                        if len(line) < 50 and line.isupper():
                            paragraph['level'] = 1
                        elif len(line) < 80 and ':' in line:
                            paragraph['level'] = 2
                        else:
                            paragraph['level'] = 1
                
                paragraphs.append(paragraph)
        
        return paragraphs

    def consolidate_paragraphs_with_llm(self, paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to merge paragraphs that should be together and fix sentence breaks."""
        if not paragraphs:
            return []
            
        if not DSPY_AVAILABLE:
            print("  • Warning: DSPy not available, skipping paragraph consolidation")
            return self._basic_paragraph_consolidation(paragraphs)
            
        try:
            # Configure DSPy with Ollama
            model_name = getattr(self, 'ollama_model', 'llama3.1:8b')
            endpoint = getattr(self, 'ollama_endpoint', 'http://localhost:11434')
            llm = dspy.LM(model=f"ollama/{model_name}", api_base=endpoint, temperature=0.0, max_tokens=200)
            dspy.settings.configure(lm=llm)
        except Exception as e:
            print(f"  • Warning: Failed to configure LLM for consolidation: {e}")
            return self._basic_paragraph_consolidation(paragraphs)
        
        # Define DSPy signature for paragraph merging
        class ParagraphMerger(dspy.Signature):
            current_paragraph = dspy.InputField(desc="Current paragraph text")
            next_paragraph = dspy.InputField(desc="Next paragraph text")
            decision = dspy.OutputField(desc="Answer 'MERGE' if paragraphs should be combined or 'SEPARATE' if they should stay separate")
            reason = dspy.OutputField(desc="Brief reason for the decision")
        
        merger = dspy.Predict(ParagraphMerger)
        consolidated = []
        current_merged = None
        batch_size = getattr(self, 'batch_size', 10)
        
        print(f"  • Consolidating {len(paragraphs)} paragraphs with LLM...")
        
        # Process paragraphs in batches
        for i in tqdm(range(0, len(paragraphs), batch_size), desc="Consolidating Paragraphs"):
            batch_paragraphs = paragraphs[i:i + batch_size]
            
            for j, para in enumerate(batch_paragraphs):
                para_idx = i + j
                para_content = para.get('content', '').strip()
                
                if not para_content:
                    continue
                    
                # If this is the first paragraph or we don't have a current merged paragraph
                if current_merged is None:
                    current_merged = para.copy()
                    continue
                
                # Check if current paragraph should be merged with the next
                should_merge = False
                reasoning = ""
                
                # Basic heuristics for obvious merges
                if self._should_merge_basic_heuristics(current_merged.get('content', ''), para_content):
                    should_merge = True
                    reasoning = "Basic heuristics suggest merge"
                else:
                    # Use LLM for more complex decisions
                    try:
                        current_text = current_merged.get('content', '')[:500]  # Limit length
                        next_text = para_content[:500]  # Limit length
                        
                        result = merger(
                            current_paragraph=current_text,
                            next_paragraph=next_text
                        )
                        
                        if result.decision.strip().upper() == 'MERGE':
                            should_merge = True
                            reasoning = result.reason
                        
                    except Exception as e:
                        print(f"    LLM merge analysis failed for paragraph {para_idx}: {e}")
                        # Fallback to basic heuristics
                        if len(current_merged.get('content', '')) < 200:  # Short paragraphs likely need merging
                            should_merge = True
                            reasoning = "Fallback: short paragraph likely continues"
                
                if should_merge:
                    # Merge paragraphs
                    current_content = current_merged.get('content', '')
                    new_content = para_content
                    
                    # Smart joining - add space or keep existing spacing
                    if current_content.endswith(('.', '!', '?', ':', ';')):
                        merged_content = f"{current_content} {new_content}"
                    elif current_content.endswith(','):
                        merged_content = f"{current_content} {new_content}"
                    else:
                        # Likely mid-sentence, join without extra space
                        merged_content = f"{current_content} {new_content}"
                    
                    current_merged['content'] = merged_content
                    # Keep the type of the first paragraph unless second is more specific
                    if para.get('type') == 'header' and current_merged.get('type') != 'header':
                        current_merged['type'] = 'header'
                        current_merged['level'] = para.get('level', 0)
                else:
                    # Don't merge - save current and start new
                    consolidated.append(current_merged)
                    current_merged = para.copy()
        
        # Don't forget the last merged paragraph
        if current_merged is not None:
            consolidated.append(current_merged)
        
        print(f"  • Consolidated from {len(paragraphs)} to {len(consolidated)} paragraphs")
        return consolidated
    
    def _should_merge_basic_heuristics(self, current: str, next_para: str) -> bool:
        """Basic heuristics to determine if paragraphs should be merged."""
        if not current or not next_para:
            return False
        
        current = current.strip()
        next_para = next_para.strip()
        
        # Don't merge if either is a clear header
        current_type = self.detect_paragraph_type(current)
        next_type = self.detect_paragraph_type(next_para)
        
        if current_type == 'header' or next_type == 'header':
            return False
        
        # Merge if current paragraph doesn't end with sentence-ending punctuation
        if not current.endswith(('.', '!', '?')):
            return True
            
        # Merge if current is very short (likely incomplete)
        if len(current) < 50:
            return True
            
        # Merge if next paragraph starts with lowercase (likely continuation)
        if next_para and next_para[0].islower():
            return True
            
        # Merge if current ends with comma or semicolon (likely list or continuation)
        if current.endswith((',', ';', ':')):
            return True
            
        return False
    
    def _basic_paragraph_consolidation(self, paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Basic paragraph consolidation when LLM is not available."""
        if not paragraphs:
            return []
            
        consolidated = []
        current_merged = None
        
        for para in paragraphs:
            para_content = para.get('content', '').strip()
            
            if not para_content:
                continue
                
            if current_merged is None:
                current_merged = para.copy()
                continue
            
            # Use only basic heuristics
            if self._should_merge_basic_heuristics(current_merged.get('content', ''), para_content):
                # Merge paragraphs
                current_content = current_merged.get('content', '')
                if current_content.endswith(('.', '!', '?', ':', ';')):
                    merged_content = f"{current_content} {para_content}"
                else:
                    merged_content = f"{current_content} {para_content}"
                
                current_merged['content'] = merged_content
            else:
                # Don't merge - save current and start new
                consolidated.append(current_merged)
                current_merged = para.copy()
        
        if current_merged is not None:
            consolidated.append(current_merged)
            
        return consolidated

    def detect_paragraph_type(self, content: str) -> str:
        """Detect the type of paragraph based on content."""
        if not content:
            return 'empty'
        
        content = content.strip()
        
        # Check for headers
        if content.startswith('#'):
            return 'header'
        
        # Check for typical discharge note headers
        header_patterns = [
            r'^[A-Z][A-Z\s]{3,}:$',  # ALL CAPS headers ending with colon (case sensitive)
            r'^[A-Z\s]{4,}$',        # ALL CAPS headers (case sensitive, 4+ chars)
            r'^[A-Z][a-z\s]+:$',     # Title Case headers ending with colon
            r'^Date\s*of\s*(Birth|Admission|Discharge)',
            r'^Name\s*:',
            r'^Unit\s*No\s*:',
            r'^Admission\s*Date',
            r'^Discharge\s*Date',
            r'^Service\s*:',
            r'^Allergies\s*:',
            r'^Attending\s*:',
            r'^Chief\s*Complaint',
            r'^History\s*of\s*Present\s*Illness',
            r'^Past\s*Medical\s*History',
            r'^Social\s*History',
            r'^Family\s*History',
            r'^Physical\s*Exam',
            r'^Pertinent\s*Results',
            r'^Assessment\s*and\s*Plan',
            r'^Medications\s*on\s*Admission',
            r'^Discharge\s*Medications',
            r'^Discharge\s*Disposition',
            r'^Discharge\s*Condition',
            r'^Discharge\s*Instructions',
            r'^Followup\s*Instructions'
        ]
        
        # Check the first 3 patterns without IGNORECASE (they need to be case-sensitive)
        case_sensitive_patterns = header_patterns[:3]
        case_insensitive_patterns = header_patterns[3:]
        
        for pattern in case_sensitive_patterns:
            if re.match(pattern, content):  # No re.IGNORECASE
                return 'header'
        
        for pattern in case_insensitive_patterns:
            if re.match(pattern, content, re.IGNORECASE):
                return 'header'
        
        # Check for lists
        if (content.startswith('- ') or content.startswith('* ') or 
            content.startswith('+ ') or re.match(r'^\d+\.', content)):
            return 'list'
        
        # Check for deidentified data patterns
        if re.search(r'___+|_\d+_|\[.*\]', content):
            return 'deidentified'
        
        # Default to text
        return 'text'

    def create_sliding_windows(self, paragraphs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Create sliding windows of 5 paragraphs: current + 4 previous."""
        if not paragraphs:
            return []
        
        windows = []
        for i in range(len(paragraphs)):
            # Create window with current paragraph and up to 4 previous paragraphs
            start_idx = max(0, i - 4)
            window = paragraphs[start_idx:i + 1]
            windows.append(window)
        
        return windows

    def llm_for_context_chunking(self, windows: List[List[Dict[str, Any]]], patient_id: str) -> List[Dict[str, Any]]:
        """Use LLM to analyze windows and determine chunk boundaries."""
        if not DSPY_AVAILABLE:
            print("  • Warning: DSPy not available, falling back to basic chunking")
            return self._basic_chunking_fallback(windows, patient_id)
        
        try:
            # Configure DSPy with Ollama
            model_name = getattr(self, 'ollama_model', 'llama3.1:8b')
            endpoint = getattr(self, 'ollama_endpoint', 'http://localhost:11434')
            llm = dspy.LM(model=f"ollama/{model_name}", api_base=endpoint, temperature=0.0, max_tokens=150)
            dspy.settings.configure(lm=llm)
        except Exception as e:
            print(f"  • Warning: Failed to configure LLM: {e}. Falling back to basic chunking")
            return self._basic_chunking_fallback(windows, patient_id)
        
        # Define DSPy signatures
        class TopicAnalyzer(dspy.Signature):
            context = dspy.InputField(desc="Previous paragraphs for context")
            current_text = dspy.InputField(desc="Current paragraph to analyze")
            current_chunk = dspy.InputField(desc="Current chunk number")
            decision = dspy.OutputField(desc="Answer 'NEW' to start new chunk or 'CONTINUE' to stay in current chunk")
            reason = dspy.OutputField(desc="Brief reason for decision")
        
        class ChunkTitleGenerator(dspy.Signature):
            chunk_content = dspy.InputField(desc="Content of the chunk")
            title = dspy.OutputField(desc="Short, descriptive title for the chunk (max 8 words)")
            description = dspy.OutputField(desc="Brief description of what this chunk covers (max 15 words)")
        
        analyzer = dspy.Predict(TopicAnalyzer)
        title_generator = dspy.Predict(ChunkTitleGenerator)
        
        enhanced_paragraphs = []
        current_chunk = 1
        current_chunk_title = f"Patient {patient_id} - Introduction"
        current_chunk_description = "Initial content"
        current_chunk_paragraphs = []
        processed_indices = set()
        
        # Get all paragraphs from windows
        all_paragraphs = []
        for window in windows:
            if window:
                current_para = window[-1]  # Last paragraph is the current one
                para_idx = len(all_paragraphs)
                if para_idx not in processed_indices:
                    all_paragraphs.append(current_para)
                    processed_indices.add(para_idx)
        
        # Process paragraphs with batching
        batch_size = getattr(self, 'batch_size', 10)
        for i in tqdm(range(0, len(windows), batch_size), desc="Semantic Chunking"):
            batch_windows = windows[i:i + batch_size]
            
            for j, window in enumerate(batch_windows):
                if not window:
                    continue
                
                window_idx = i + j
                current_para = window[-1]  # Current paragraph is the last in window
                
                should_create_new = False
                reasoning = ""
                
                # Always start new chunk on headers
                if current_para.get('type') == 'header':
                    should_create_new = True
                    reasoning = "Header starts a new section."
                
                # For non-headers, use LLM analysis if we have context
                elif len(window) > 1:
                    context_paragraphs = window[:-1]  # All except current
                    context_str = "\n".join([p.get('content', '') for p in context_paragraphs])
                    current_text = current_para.get('content', '')
                    
                    try:
                        result = analyzer(
                            context=context_str[:800],  # Limit context length
                            current_text=current_text[:400],  # Limit current text length
                            current_chunk=str(current_chunk)
                        )
                        if result.decision.strip().upper() == 'NEW':
                            should_create_new = True
                            reasoning = result.reason
                    except Exception as e:
                        print(f"    LLM analysis failed for paragraph {window_idx}: {e}")
                        # Fallback: create new chunk every 6 paragraphs
                        if len(current_chunk_paragraphs) >= 6:
                            should_create_new = True
                            reasoning = "Fallback: maximum paragraphs per chunk reached"
                
                # Create new chunk if needed
                if should_create_new and current_chunk_paragraphs:
                    current_chunk += 1
                    
                    # Generate title for completed chunk
                    chunk_content = " ".join([p.get('content', '') for p in current_chunk_paragraphs])
                    try:
                        title_res = title_generator(chunk_content=chunk_content[:1000])
                        current_chunk_title = title_res.title
                        current_chunk_description = title_res.description
                    except Exception as e:
                        print(f"    Title generation failed: {e}")
                        current_chunk_title = f"Patient {patient_id} - Chunk {current_chunk}"
                        current_chunk_description = "Medical record section"
                
                # Add paragraph to enhanced list with chunk info
                para_enhanced = current_para.copy()
                para_enhanced['chunk_number'] = current_chunk
                para_enhanced['chunk_title'] = current_chunk_title
                para_enhanced['chunk_description'] = current_chunk_description
                # Clean escape characters from content
                para_enhanced['content'] = self.remove_escape_characters_final(para_enhanced['content'])
                
                enhanced_paragraphs.append(para_enhanced)
                current_chunk_paragraphs.append(para_enhanced)
                
                # Start new chunk collection if we just created one
                if should_create_new:
                    current_chunk_paragraphs = [para_enhanced]
        
        return enhanced_paragraphs
    
    def _basic_chunking_fallback(self, windows: List[List[Dict[str, Any]]], patient_id: str) -> List[Dict[str, Any]]:
        """Fallback chunking when LLM is not available."""
        enhanced_paragraphs = []
        current_chunk = 1
        
        # Get all paragraphs from windows
        all_paragraphs = []
        processed_contents = set()
        
        for window in windows:
            if window:
                current_para = window[-1]
                content = current_para.get('content', '')
                if content not in processed_contents:
                    all_paragraphs.append(current_para)
                    processed_contents.add(content)
        
        # Basic chunking: every 5 paragraphs or on headers
        for i, para in enumerate(all_paragraphs):
            if i > 0 and (i % 5 == 0 or para.get('type') == 'header'):
                current_chunk += 1
            
            para_enhanced = para.copy()
            para_enhanced['chunk_number'] = current_chunk
            para_enhanced['chunk_title'] = f"Patient {patient_id} - Chunk {current_chunk}"
            para_enhanced['chunk_description'] = "Medical record section"
            # Clean escape characters from content
            para_enhanced['content'] = self.remove_escape_characters_final(para_enhanced['content'])
            
            enhanced_paragraphs.append(para_enhanced)
        
        return enhanced_paragraphs

    def convert_chunks_to_json(self, paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert enhanced paragraphs to chunk format."""
        chunks_map = defaultdict(lambda: {"paragraphs": []})
        
        for para in paragraphs:
            chunk_num = para.get('chunk_number', 0)
            if "title" not in chunks_map[chunk_num]:
                chunks_map[chunk_num]['chunk_number'] = chunk_num
                chunks_map[chunk_num]['title'] = para.get('chunk_title', 'Untitled')
                chunks_map[chunk_num]['description'] = para.get('chunk_description', 'No description')
            
            para_obj = {
                "content": para.get('content', ''),
                "type": para.get('type', 'text'),
                "level": para.get('level', 0)
            }
            chunks_map[chunk_num]["paragraphs"].append(para_obj)
        
        return list(chunks_map.values())

    def process_patient_texts(self, df: pd.DataFrame, patient_ids: List[int]) -> Dict[str, Any]:
        """Process discharge texts for specified patients."""
        all_chunks = []
        total_paragraphs = 0
        
        for patient_id in patient_ids:
            print(f"  • Processing patient {patient_id}...")
            
            # Get all discharge notes for this patient
            patient_notes = df[df['subject_id'] == patient_id]
            
            if patient_notes.empty:
                print(f"    No notes found for patient {patient_id}")
                continue
            
            # Combine all text for this patient
            combined_text = ""
            for _, note in patient_notes.iterrows():
                note_text = self.clean_text(note['text'])
                if note_text:
                    combined_text += f"\n\n--- Note ID: {note['note_id']} ---\n\n"
                    combined_text += note_text
            
            if not combined_text.strip():
                print(f"    No valid text found for patient {patient_id}")
                continue
            
            # Split into paragraphs
            paragraphs = self.split_into_paragraphs(combined_text)
            print(f"    Found {len(paragraphs)} initial paragraphs")
            
            # Consolidate paragraphs using LLM to merge related content and fix sentence breaks
            consolidated_paragraphs = self.consolidate_paragraphs_with_llm(paragraphs)
            total_paragraphs += len(consolidated_paragraphs)
            
            print(f"    Consolidated to {len(consolidated_paragraphs)} paragraphs")
            
            # Create sliding windows
            windows = self.create_sliding_windows(consolidated_paragraphs)
            print(f"    Created {len(windows)} sliding windows")
            
            # Apply LLM-based semantic chunking
            enhanced_paragraphs = self.llm_for_context_chunking(windows, str(patient_id))
            
            # Convert to chunk format
            patient_chunks = self.convert_chunks_to_json(enhanced_paragraphs)
            all_chunks.extend(patient_chunks)
            
            print(f"    Created {len(patient_chunks)} semantic chunks")
        
        # Create the final document structure
        document = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_paragraphs": total_paragraphs,
                "total_chunks": len(all_chunks),
                "source": "MIMIC discharge notes",
                "patients_processed": len(patient_ids),
                "patient_ids": [int(pid) for pid in patient_ids]
            },
            "chunks": all_chunks
        }
        
        return document

def load_mimic_data(file_path: str = "note/discharge.csv.gz", n_patients: int = 10) -> tuple:
    """Load MIMIC discharge data and get first n unique patients."""
    print(f"Loading MIMIC data from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path, compression='gzip')
    print(f"  • Loaded {len(df)} discharge notes")
    
    # Get unique patient IDs
    unique_patients = df['subject_id'].unique()
    print(f"  • Found {len(unique_patients)} unique patients")
    
    # Get first n patients
    selected_patients = sorted(unique_patients)[:n_patients]
    print(f"  • Selected first {n_patients} patients: {selected_patients}")
    
    # Filter data for selected patients
    filtered_df = df[df['subject_id'].isin(selected_patients)]
    print(f"  • Filtered to {len(filtered_df)} notes for selected patients")
    
    return filtered_df, selected_patients

def main():
    """Main function to process MIMIC discharge notes."""
    parser = argparse.ArgumentParser(
        description="Process MIMIC discharge notes into chunked JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_mimic.py
  python process_mimic.py --output-file mimic_chunked.json
  python process_mimic.py --n-patients 5
        """
    )
    
    parser.add_argument("--output-file", "-o", 
                       default="mimic_discharge_chunked.json",
                       help="Output JSON file (default: mimic_discharge_chunked.json)")
    parser.add_argument("--n-patients", "-n", type=int, default=10,
                       help="Number of unique patients to process (default: 10)")
    parser.add_argument("--input-file", "-i", 
                       default="note/discharge.csv.gz",
                       help="Input discharge CSV file (default: note/discharge.csv.gz)")
    parser.add_argument("--ollama-model", default="llama3.1:8b",
                       help="Ollama model for semantic chunking (default: llama3.1:8b)")
    parser.add_argument("--ollama-endpoint", default="http://localhost:11434",
                       help="Ollama endpoint (default: http://localhost:11434)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for LLM processing (default: 10)")
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("MIMIC DISCHARGE NOTES PROCESSING PIPELINE")
        print("=" * 60)
        
        # Load data
        print("\nSTEP 1: Loading MIMIC Data")
        print("-" * 40)
        df, patient_ids = load_mimic_data(args.input_file, args.n_patients)
        
        # Process text
        print("\nSTEP 2: Processing Discharge Texts")
        print("-" * 40)
        processor = MimicTextProcessor()
        # Configure processor with command line arguments
        processor.ollama_model = args.ollama_model
        processor.ollama_endpoint = args.ollama_endpoint
        processor.batch_size = args.batch_size
        
        print(f"  • Using LLM: {args.ollama_model} at {args.ollama_endpoint}")
        print(f"  • Batch size: {args.batch_size}")
        
        document = processor.process_patient_texts(df, patient_ids)
        
        # Save output
        print("\nSTEP 3: Saving Output")
        print("-" * 40)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, ensure_ascii=False)
        
        print(f"  • Output saved to: {args.output_file}")
        print(f"  • Processed {document['metadata']['patients_processed']} patients")
        print(f"  • Created {document['metadata']['total_chunks']} chunks")
        print(f"  • Total paragraphs: {document['metadata']['total_paragraphs']}")
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nError in processing: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)