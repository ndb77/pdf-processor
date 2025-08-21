# pdf-processor
Implementation of Wu et. al paper

## Chunking
1. Chunking must use a venv for chunking. Install packages into the venv using requirements-unified.txt


## Entity processing
1. Entity processing must use a mamba environment. Install packages using the mamba environment. Install nmslib using the instructions found in requirements_ep.txt
2. a joined.txt file must be created before running entitiy_processor_improved.py
* create this txt by running semantic_types_definitions_and_cui.py and point it at your UMLS <MRCONSO.RRF> <MRSTY.RRF> <MRDEF.RRF> files with the output being <b>joined.txt</b>
3. You must have the MIMIC4 dataset accessible. Use <b>process_mimic.py</b> to generate a chunked document from the mimic4 notes(discharge_detail, discharge, radiology_detail, radiology). 
4. You must have ollama llama3.1:8b installed and serving at localhost:11434
