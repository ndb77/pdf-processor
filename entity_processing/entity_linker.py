#!/usr/bin/env python3
"""
Entity Linker

This script processes a folder containing *entity_processing_results.json files
and creates duplicates with new naming convention for further processing.

The output files are named as [first-five-characters-of-document-name]_linked_entities.json
"""

import json
import os
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
import numpy as np
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings


@dataclass
class CorpusEntity:
    """Represents an entity from a corpus document with its embedding."""
    entity_name: str
    cui: str
    entity_type: str
    entity_description: str
    umls_definition: str
    source_document: str
    embedding: np.ndarray
    paragraph_index: int
    chunk_id: str


@dataclass
class SimilarEntity:
    """Represents a similar entity reference."""
    entity_name: str
    cui: str
    entity_type: str
    entity_description: str
    umls_definition: str
    source_document: str
    similarity_score: float


# DEPRECATED: Numpy-based cosine similarity calculation (replaced by ChromaDB)
# def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
#     """
#     Calculate cosine similarity between two vectors.
#     
#     Args:
#         vec1: First vector
#         vec2: Second vector
#         
#     Returns:
#         Cosine similarity score between -1 and 1
#     """
#     try:
#         # Validate input vectors
#         if vec1.size == 0 or vec2.size == 0:
#             return 0.0
#         
#         if vec1.shape != vec2.shape:
#             return 0.0
#         
#         # Normalize vectors
#         norm1 = np.linalg.norm(vec1)
#         norm2 = np.linalg.norm(vec2)
#         
#         if norm1 == 0 or norm2 == 0:
#             return 0.0
#         
#         # Calculate cosine similarity (can range from -1 to 1)
#         similarity = np.dot(vec1, vec2) / (norm1 * norm2)
#         
#         # Clamp to handle floating point precision errors
#         return max(-1.0, min(1.0, float(similarity)))
#         
#     except Exception:
#         return 0.0


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class EntityLinker:
    """
    Processes entity processing results files and creates linked entity duplicates.
    """
    
    def __init__(self, input_folder: str, output_folder: Optional[str] = None, 
                 enable_similarity_search: bool = False,
                 similarity_threshold: float = 0.5, batch_size: int = 1000,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the EntityLinker.
        
        Args:
            input_folder: Path to folder containing *entity_processing_results.json files
            output_folder: Optional output folder. If None, uses 'linked' subfolder in input_folder
            enable_similarity_search: Enable cross-document similarity search using ChromaDB
            similarity_threshold: Minimum similarity score threshold (adjustable)
            batch_size: Number of entities to process in batches for ChromaDB
            logger: Optional logger instance
        """
        self.input_folder = Path(input_folder)
        # Default to 'linked' subfolder within input directory
        self.output_folder = Path(output_folder) if output_folder else self.input_folder / "linked"
        self.enable_similarity_search = enable_similarity_search
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.logger = logger or setup_logging()
        
        # Initialize ChromaDB client and collections
        self.chroma_client = None
        self.corpus_collection = None
        self.rag_collection = None
        self.corpus_loaded = False
        
        if self.enable_similarity_search:
            try:
                # Initialize in-memory ChromaDB client
                self.chroma_client = chromadb.Client()
                
                # Create separate collections for corpus and rag entities with cosine similarity
                self.corpus_collection = self.chroma_client.create_collection(
                    name="corpus_entities",
                    metadata={"description": "Entities from corpus documents"},
                    configuration={"hnsw": {"space": "cosine"}}
                )
                
                self.rag_collection = self.chroma_client.create_collection(
                    name="rag_entities", 
                    metadata={"description": "Entities from rag documents"},
                    configuration={"hnsw": {"space": "cosine"}}
                )
                
                self.logger.info("ChromaDB collections initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize ChromaDB: {e}")
                self.enable_similarity_search = False
                self.logger.warning("Similarity search disabled due to ChromaDB initialization failure")
        
        # Initialize legacy corpus entities cache (for fallback)
        self.corpus_entities: List[CorpusEntity] = []
        
        # Validate input folder
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder does not exist: {self.input_folder}")
        if not self.input_folder.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_folder}")
    
    def find_entity_files(self) -> List[Path]:
        """
        Find all *entity_processing_results.json files in the input folder.
        
        Returns:
            List of Path objects for found files
        """
        pattern = "*entity_processing_results.json"
        files = list(self.input_folder.glob(pattern))
        
        self.logger.info(f"Found {len(files)} files matching pattern '{pattern}'")
        for file in files:
            self.logger.debug(f"Found file: {file.name}")
        
        return files
    
    def load_corpus_entities(self) -> None:
        """
        Load all entities from corpus-document files into ChromaDB collections for similarity search.
        """
        if self.corpus_loaded:
            return
            
        if not self.enable_similarity_search or not self.corpus_collection:
            self.logger.warning("ChromaDB not initialized, skipping corpus entity loading")
            return
        
        self.logger.info("Loading corpus entities into ChromaDB collections...")
        
        files = self.find_entity_files()
        corpus_files = []
        rag_files = []
        
        # First pass: identify document files by type
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                document_type = data.get("metadata", {}).get("document_type")
                if document_type == "corpus-document":
                    corpus_files.append((file, data))
                elif document_type == "rag-document":
                    rag_files.append((file, data))
                    
            except Exception as e:
                self.logger.warning(f"Error reading file {file.name}: {e}")
                continue
        
        self.logger.info(f"Found {len(corpus_files)} corpus-document files and {len(rag_files)} rag-document files")
        
        # Second pass: batch load entities into ChromaDB collections
        total_corpus_entities = 0
        total_rag_entities = 0
        
        # Load corpus entities
        for file_path, json_data in corpus_files:
            entities_count = self._load_entities_to_chromadb(json_data, file_path.name, "corpus")
            total_corpus_entities += entities_count
        
        # Load rag entities  
        for file_path, json_data in rag_files:
            entities_count = self._load_entities_to_chromadb(json_data, file_path.name, "rag")
            total_rag_entities += entities_count
        
        self.corpus_loaded = True
        self.logger.info(f"Loaded {total_corpus_entities} corpus entities and {total_rag_entities} rag entities into ChromaDB")
    
    def _load_entities_to_chromadb(self, json_data: Dict[str, Any], source_file: str, collection_type: str) -> int:
        """
        Load entities from a JSON document into ChromaDB collections in batches.
        
        Args:
            json_data: The JSON data
            source_file: Name of the source file
            collection_type: Type of collection ("corpus" or "rag")
            
        Returns:
            Number of entities loaded
        """
        entities_count = 0
        batch_embeddings = []
        batch_metadatas = []
        batch_ids = []
        
        # Select appropriate collection
        target_collection = self.corpus_collection if collection_type == "corpus" else self.rag_collection
        
        if not target_collection:
            self.logger.error(f"No {collection_type} collection available")
            return 0
        
        try:
            chunks = json_data.get("chunks", {})
            
            for chunk_id, chunk_data in chunks.items():
                paragraphs = chunk_data.get("paragraphs", [])
                
                for paragraph in paragraphs:
                    paragraph_index = paragraph.get("paragraph_index", 0)
                    entities = paragraph.get("entities", [])
                    
                    for entity in entities:
                        # Extract required fields
                        entity_name = entity.get("entity_name", "")
                        cui = entity.get("cui", "")
                        entity_type = entity.get("entity_type", "")
                        entity_description = entity.get("entity_description", "")
                        umls_definition = entity.get("umls_definition", "")
                        content_embedding = entity.get("content_embedding", [])
                        
                        # Validate embedding
                        if not content_embedding or not isinstance(content_embedding, list):
                            continue
                        
                        try:
                            # Validate embedding array
                            embedding_array = np.array(content_embedding, dtype=np.float32)
                            if embedding_array.size == 0:
                                continue
                            
                            # Generate unique ID for this entity
                            entity_id = f"{source_file}_{chunk_id}_{paragraph_index}_{entities_count}"
                            
                            # Prepare metadata for ChromaDB
                            metadata = {
                                "entity_name": entity_name,
                                "entity_type": entity_type,
                                "entity_description": entity_description,
                                "umls_definition": umls_definition,
                                "cui": cui,
                                "source_document": source_file,
                                "paragraph_index": paragraph_index,
                                "chunk_id": str(chunk_id)
                            }
                            
                            # Add to batch
                            batch_embeddings.append(embedding_array.tolist())
                            batch_metadatas.append(metadata)
                            batch_ids.append(entity_id)
                            entities_count += 1
                            
                            # Process batch when it reaches batch_size
                            if len(batch_embeddings) >= self.batch_size:
                                self._add_batch_to_chromadb(target_collection, batch_embeddings, 
                                                          batch_metadatas, batch_ids)
                                batch_embeddings.clear()
                                batch_metadatas.clear()
                                batch_ids.clear()
                            
                        except Exception as e:
                            self.logger.debug(f"Error processing entity embedding: {e}")
                            continue
            
            # Process remaining entities in batch
            if batch_embeddings:
                self._add_batch_to_chromadb(target_collection, batch_embeddings, 
                                          batch_metadatas, batch_ids)
                            
        except Exception as e:
            self.logger.error(f"Error loading entities from {source_file}: {e}")
        
        return entities_count
    
    def _add_batch_to_chromadb(self, collection, embeddings: List[List[float]], 
                              metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """
        Add a batch of entities to ChromaDB collection.
        
        Args:
            collection: ChromaDB collection object
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of unique IDs
        """
        try:
            collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            self.logger.debug(f"Added batch of {len(embeddings)} entities to ChromaDB")
            
        except Exception as e:
            self.logger.error(f"Error adding batch to ChromaDB: {e}")
    
    def find_similar_entities(self, query_embedding: np.ndarray, query_cui: str = "", 
                             query_source: str = "") -> List[SimilarEntity]:
        """
        Find the most similar entities to the query embedding using ChromaDB.
        
        Args:
            query_embedding: The embedding vector to search for
            query_cui: CUI of the query entity (for self-filtering)
            query_source: Source document of query entity (for self-filtering)
            
        Returns:
            List of similar entities above the similarity threshold
        """
        if not self.enable_similarity_search or not self.corpus_collection:
            self.logger.debug("ChromaDB similarity search not available")
            return []
        
        try:
            # Query ChromaDB corpus collection for similar entities
            # Use a large n_results to get all entities, then filter by threshold
            query_results = self.corpus_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(self.corpus_collection.count(), 1000),  # Get up to 1000 results
                include=['metadatas', 'distances']
            )
            
            similarities = []
            skipped_self = 0
            below_threshold = 0
            
            if query_results and query_results['metadatas'] and query_results['distances']:
                metadatas = query_results['metadatas'][0]  # First (only) query result
                distances = query_results['distances'][0]   # First (only) query result
                
                for metadata, distance in zip(metadatas, distances):
                    # Skip self-references (same CUI)
                    if query_cui and metadata.get("cui") == query_cui:
                        skipped_self += 1
                        continue
                    
                    # Convert ChromaDB cosine distance to cosine similarity
                    # For cosine distance: distance = 1.0 - cosine_similarity
                    # So: cosine_similarity = 1.0 - distance (range 0.0 to 2.0)
                    similarity_score = 1.0 - distance
                    
                    # Validate distance is in expected range for cosine distance
                    if distance < 0.0 or distance > 2.0:
                        self.logger.warning(f"Unexpected cosine distance value: {distance} (expected 0.0-2.0)")
                    
                    # Apply threshold filter
                    if similarity_score >= self.similarity_threshold:
                        similar_entity = SimilarEntity(
                            entity_name=metadata.get("entity_name", ""),
                            cui=metadata.get("cui", ""),
                            entity_type=metadata.get("entity_type", ""),
                            entity_description=metadata.get("entity_description", ""),
                            umls_definition=metadata.get("umls_definition", ""),
                            source_document=metadata.get("source_document", ""),
                            similarity_score=similarity_score
                        )
                        similarities.append(similar_entity)
                    else:
                        below_threshold += 1
            
            # Sort by absolute similarity score (descending)
            similarities.sort(key=lambda x: abs(x.similarity_score), reverse=True)
            
            # Enhanced logging with sample scores
            total_results = len(metadatas) if query_results and query_results['metadatas'] else 0
            self.logger.debug(f"ChromaDB search: {total_results} retrieved, {skipped_self} self-skipped, "
                             f"{below_threshold} below threshold, {len(similarities)} meeting threshold")
            
            if similarities:
                self.logger.debug(f"Found {len(similarities)} entities meeting threshold (>= {self.similarity_threshold})")
                # Log top 3 similarity scores for debugging
                for i, sim in enumerate(similarities[:3]):
                    self.logger.debug(f"  Top {i+1}: {sim.entity_name} (CUI: {sim.cui}) - Score: {sim.similarity_score:.4f}")
            else:
                self.logger.debug(f"No entities found above similarity threshold {self.similarity_threshold}")
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"Error in ChromaDB similarity search: {e}")
            return []
    
    def extract_document_name(self, json_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract document name from JSON metadata.
        
        Args:
            json_data: Loaded JSON data
            
        Returns:
            Document name if found, None otherwise
        """
        try:
            metadata = json_data.get("metadata", {})
            document_name = metadata.get("document_name")
            
            if not document_name:
                self.logger.warning("No document_name found in metadata")
                return None
            
            self.logger.debug(f"Extracted document name: {document_name}")
            return document_name
            
        except Exception as e:
            self.logger.error(f"Error extracting document name: {e}")
            return None
    
    def add_reference_of_field(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add 'reference_of' field to each entity in the JSON data.
        For rag-document types, populate with similar entities from corpus.
        
        Args:
            json_data: The loaded JSON data
            
        Returns:
            Modified JSON data with reference_of fields added
        """
        try:
            # Deep copy to avoid modifying the original
            import copy
            modified_data = copy.deepcopy(json_data)
            
            # Check document type
            document_type = modified_data.get("metadata", {}).get("document_type", "")
            is_rag_document = document_type == "rag-document"
            
            # Load corpus entities if similarity search is enabled and this is a rag document
            if self.enable_similarity_search and is_rag_document:
                if not self.corpus_loaded:
                    self.load_corpus_entities()
            
            # Navigate through chunks and paragraphs to find entities
            chunks = modified_data.get("chunks", {})
            entities_processed = 0
            similarity_searches = 0
            
            for chunk_id, chunk_data in chunks.items():
                paragraphs = chunk_data.get("paragraphs", [])
                
                for paragraph in paragraphs:
                    entities = paragraph.get("entities", [])
                    
                    for entity in entities:
                        if is_rag_document and self.enable_similarity_search:
                            # Perform similarity search for rag-document entities
                            content_embedding = entity.get("content_embedding", [])
                            
                            if content_embedding and isinstance(content_embedding, list):
                                try:
                                    query_embedding = np.array(content_embedding, dtype=np.float32)
                                    
                                    # Validate embedding
                                    if query_embedding.size == 0:
                                        self.logger.debug("Empty embedding for entity")
                                        entity["reference_of"] = []
                                        continue
                                    
                                    # Get entity info for self-filtering
                                    query_cui = entity.get("cui", "")
                                    document_name = modified_data.get("metadata", {}).get("document_name", "")
                                    
                                    # Perform similarity search
                                    similar_entities = self.find_similar_entities(
                                        query_embedding, 
                                        query_cui=query_cui, 
                                        query_source=document_name
                                    )
                                    
                                    # Format similar entities as list of dictionaries
                                    reference_list = []
                                    for sim_entity in similar_entities:
                                        reference_list.append({
                                            "entity_name": sim_entity.entity_name,
                                            "cui": sim_entity.cui,
                                            "entity_type": sim_entity.entity_type,
                                            "entity_description": sim_entity.entity_description,
                                            "umls_definition": sim_entity.umls_definition,
                                            "source_document": sim_entity.source_document,
                                            "similarity_score": round(sim_entity.similarity_score, 4)
                                        })
                                    
                                    entity["reference_of"] = reference_list
                                    similarity_searches += 1
                                    
                                    # Debug logging for results
                                    if reference_list:
                                        self.logger.debug(f"Entity '{entity.get('entity_name', 'unknown')}' found {len(reference_list)} similar entities")
                                        if len(reference_list) > 0:
                                            top_match = reference_list[0]
                                            self.logger.debug(f"  Top match: {top_match['entity_name']} (Score: {top_match['similarity_score']})")
                                    else:
                                        self.logger.debug(f"No similar entities found for entity '{entity.get('entity_name', 'unknown')}' "
                                                        f"(CUI: {query_cui}) with threshold {self.similarity_threshold}")
                                    
                                except Exception as e:
                                    self.logger.debug(f"Error in similarity search for entity '{entity.get('entity_name', 'unknown')}': {e}")
                                    entity["reference_of"] = []
                            else:
                                entity["reference_of"] = []
                        else:
                            # For corpus documents or when similarity search is disabled
                            entity["reference_of"] = []
                        
                        entities_processed += 1
            
            if is_rag_document and self.enable_similarity_search:
                self.logger.debug(f"Performed similarity search for {similarity_searches}/{entities_processed} entities")
            else:
                self.logger.debug(f"Added empty 'reference_of' field to {entities_processed} entities")
            
            return modified_data
            
        except Exception as e:
            self.logger.error(f"Error adding reference_of field: {e}")
            return json_data
    
    def generate_output_filename(self, document_name: str, original_filename: str, document_type: str = "") -> str:
        """
        Generate output filename based on document name and type.
        
        Args:
            document_name: Full document name from metadata
            original_filename: Original file name as fallback
            document_type: Document type (rag-document, corpus-document, etc.)
            
        Returns:
            Generated filename with format [first-five-chars]_[rag/corpus]_linked_entities.json
        """
        try:
            # Extract first 5 characters from document name
            # Remove common file extensions and clean the name
            clean_name = document_name.replace(".pdf", "").replace("_chunked.json", "")
            first_five = clean_name[:5]
            
            # Ensure we have valid characters for filename
            safe_chars = "".join(c for c in first_five if c.isalnum() or c in ".-_")
            
            if not safe_chars:
                # Fallback to original filename prefix
                original_base = Path(original_filename).stem
                safe_chars = original_base[:5]
                self.logger.warning(f"Using fallback naming from original file: {safe_chars}")
            
            # Determine document type suffix
            type_suffix = ""
            if document_type == "rag-document":
                type_suffix = "_rag"
            elif document_type == "corpus-document":
                type_suffix = "_corpus"
            else:
                # Fallback for unknown document types
                type_suffix = "_unknown"
                self.logger.warning(f"Unknown document type: {document_type}")
            
            output_filename = f"{safe_chars}{type_suffix}_linked_entities.json"
            self.logger.debug(f"Generated output filename: {output_filename}")
            
            return output_filename
            
        except Exception as e:
            self.logger.error(f"Error generating output filename: {e}")
            # Ultimate fallback
            return f"linked_entities_{Path(original_filename).stem}.json"
    
    def process_file(self, input_file: Path) -> bool:
        """
        Process a single entity processing results file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Processing file: {input_file.name}")
            
            # Load JSON data
            with open(input_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Add reference_of field to all entities
            json_data = self.add_reference_of_field(json_data)
            
            # Extract document name and type
            document_name = self.extract_document_name(json_data)
            document_type = json_data.get("metadata", {}).get("document_type", "")
            
            # Generate output filename
            if document_name:
                output_filename = self.generate_output_filename(document_name, input_file.name, document_type)
            else:
                # Fallback naming if no document name found
                type_suffix = "_rag" if document_type == "rag-document" else "_corpus" if document_type == "corpus-document" else "_unknown"
                output_filename = f"{input_file.stem[:5]}{type_suffix}_linked_entities.json"
                self.logger.warning(f"Using fallback filename: {output_filename}")
            
            # Create output path
            output_path = self.output_folder / output_filename
            
            # Ensure output directory exists
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            # Write duplicate file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Created linked entities file: {output_filename}")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in file {input_file.name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error processing file {input_file.name}: {e}")
            return False
    
    def process_all(self) -> Dict[str, int]:
        """
        Process all entity processing results files in the input folder.
        
        Returns:
            Dictionary with processing statistics
        """
        files = self.find_entity_files()
        
        if not files:
            self.logger.warning("No files found to process")
            return {"total": 0, "successful": 0, "failed": 0}
        
        successful = 0
        failed = 0
        
        for file in files:
            if self.process_file(file):
                successful += 1
            else:
                failed += 1
        
        stats = {
            "total": len(files),
            "successful": successful,
            "failed": failed
        }
        
        self.logger.info(f"Processing complete: {successful}/{len(files)} files successful")
        if failed > 0:
            self.logger.warning(f"{failed} files failed to process")
        
        return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process entity processing results and create linked entity files"
    )
    parser.add_argument(
        "--input-folder",
        "-i",
        required=True,
        help="Path to folder containing *entity_processing_results.json files"
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        help="Output folder (default: 'linked' subfolder in input folder)"
    )
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--enable-similarity-search",
        action="store_true",
        help="Enable cross-document similarity search for rag-document entities"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Minimum similarity score threshold (default: 0.5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for ChromaDB loading (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Create EntityLinker instance
        linker = EntityLinker(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            enable_similarity_search=args.enable_similarity_search,
            similarity_threshold=args.similarity_threshold,
            batch_size=args.batch_size,
            logger=logger
        )
        
        # Process all files
        stats = linker.process_all()
        
        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Total files found: {stats['total']}")
        print(f"Successfully processed: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        
        if stats['failed'] > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()