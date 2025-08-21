#!/usr/bin/env python3
"""
Program to join three UMLS files by CUI and create a unified output file.
Input files:
- MRCONSO.RRF: CUI (col 0) and entity name (col 14)
- MRSTY.RRF: CUI (col 0) and entity type (col 3)
- MRDEF.RRF: CUI (col 0) and entity definition (col 6)
Output format: cui|entity name|entity type|entity definition
"""

import sys
import os
from collections import defaultdict

def read_mrconso(file_path):
    """Read MRCONSO.RRF file and extract CUI and entity name for active English terms only."""
    cui_to_name = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    fields = line.strip().split('|')
                    if len(fields) >= 15:  # Ensure we have enough columns
                        cui = fields[0]
                        lat = fields[1]  # Language field
                        suppress = fields[16] if len(fields) > 16 else ''  # SUPPRESS field
                        entity_name = fields[14]
                        
                        # Filter for active English terms only
                        if lat == 'ENG' and suppress != 'O':
                            cui_to_name[cui] = entity_name
                    else:
                        print(f"Warning: Line {line_num} in MRCONSO.RRF has insufficient columns")
                except Exception as e:
                    print(f"Error processing line {line_num} in MRCONSO.RRF: {e}")
    except FileNotFoundError:
        print(f"Error: MRCONSO.RRF file not found at {file_path}")
        return {}
    except Exception as e:
        print(f"Error reading MRCONSO.RRF: {e}")
        return {}
    
    print(f"Read {len(cui_to_name)} active English entries from MRCONSO.RRF")
    return cui_to_name

def read_mrsty(file_path):
    """Read MRSTY.RRF file and extract CUI, TUI, and entity type."""
    cui_to_tui = {}
    cui_to_type = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    fields = line.strip().split('|')
                    if len(fields) >= 4:  # Ensure we have enough columns
                        cui = fields[0]
                        tui = fields[1]  # TUI number from column 1
                        entity_type = fields[3]  # Semantic type from column 3
                        cui_to_tui[cui] = tui
                        cui_to_type[cui] = entity_type
                    else:
                        print(f"Warning: Line {line_num} in MRSTY.RRF has insufficient columns")
                except Exception as e:
                    print(f"Error processing line {line_num} in MRSTY.RRF: {e}")
    except FileNotFoundError:
        print(f"Error: MRSTY.RRF file not found at {file_path}")
        return {}, {}
    except Exception as e:
        print(f"Error reading MRSTY.RRF: {e}")
        return {}, {}
    
    print(f"Read {len(cui_to_type)} entries from MRSTY.RRF")
    return cui_to_tui, cui_to_type

def read_mrdef(file_path):
    """Read MRDEF.RRF file and extract CUI and entity definition."""
    cui_to_def = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    fields = line.strip().split('|')
                    if len(fields) >= 7:  # Ensure we have enough columns
                        cui = fields[0]
                        entity_def = fields[5]
                        cui_to_def[cui] = entity_def
                    else:
                        print(f"Warning: Line {line_num} in MRDEF.RRF has insufficient columns")
                except Exception as e:
                    print(f"Error processing line {line_num} in MRDEF.RRF: {e}")
    except FileNotFoundError:
        print(f"Error: MRDEF.RRF file not found at {file_path}")
        return {}
    except Exception as e:
        print(f"Error reading MRDEF.RRF: {e}")
        return {}
    
    print(f"Read {len(cui_to_def)} entries from MRDEF.RRF")
    return cui_to_def

def join_files(mrconso_path, mrsty_path, mrdef_path, output_path):
    """Join the three files by CUI and write the output."""
    
    # Read all three files
    print("Reading MRCONSO.RRF...")
    cui_to_name = read_mrconso(mrconso_path)
    
    print("Reading MRSTY.RRF...")
    cui_to_tui, cui_to_type = read_mrsty(mrsty_path)
    
    print("Reading MRDEF.RRF...")
    cui_to_def = read_mrdef(mrdef_path)
    
    # Get all unique CUIs
    all_cuis = set(cui_to_name.keys()) | set(cui_to_type.keys()) | set(cui_to_def.keys())
    print(f"Found {len(all_cuis)} unique CUIs across all files")
    
    # Write joined output
    joined_count = 0
    missing_data_count = 0
    
    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for cui in sorted(all_cuis):
                entity_name = cui_to_name.get(cui, '')
                entity_type = cui_to_type.get(cui, '')
                entity_def = cui_to_def.get(cui, '')
                tui_number = cui_to_tui.get(cui, '')
                
                # Check if we have at least some data for this CUI
                if entity_name or entity_type or entity_def or tui_number:
                    output_line = f"{cui}|{entity_name}|{entity_type}|{entity_def}|{tui_number}"
                    output_file.write(output_line + '\n')
                    joined_count += 1
                else:
                    missing_data_count += 1
        
        print(f"Successfully wrote {joined_count} joined entries to {output_path}")
        if missing_data_count > 0:
            print(f"Warning: {missing_data_count} CUIs had no data from any file")
            
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False
    
    return True

def main():
    """Main function to handle command line arguments and execute the join."""
    
    # Check command line arguments
    if len(sys.argv) != 5:
        print("Usage: python semantic_types_definitions_and_cui.py <MRCONSO.RRF> <MRSTY.RRF> <MRDEF.RRF> <output_file>")
        print("Example: python semantic_types_definitions_and_cui.py MRCONSO.RRF MRSTY.RRF MRDEF.RRF joined_output.txt")
        sys.exit(1)
    
    mrconso_path = sys.argv[1]
    mrsty_path = sys.argv[2]
    mrdef_path = sys.argv[3]
    output_path = sys.argv[4]
    
    # Check if input files exist
    for file_path in [mrconso_path, mrsty_path, mrdef_path]:
        if not os.path.exists(file_path):
            print(f"Error: Input file {file_path} does not exist")
            sys.exit(1)
    
    print("Starting file join process...")
    print(f"MRCONSO.RRF: {mrconso_path}")
    print(f"MRSTY.RRF: {mrsty_path}")
    print(f"MRDEF.RRF: {mrdef_path}")
    print(f"Output file: {output_path}")
    print("-" * 50)
    
    # Perform the join
    success = join_files(mrconso_path, mrsty_path, mrdef_path, output_path)
    
    if success:
        print("\nJoin completed successfully!")
        print(f"Output file: {output_path}")
    else:
        print("\nJoin failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
