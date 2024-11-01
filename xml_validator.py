# xml_validator.py

import xml.etree.ElementTree as ET
from pathlib import Path
import sys

def validate_xml_file(file_path):
    """Validate XML file and print summary of contents."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        print(f"\nSuccessfully parsed {file_path}")
        print(f"Root tag: {root.tag}")
        print(f"Number of child elements: {len(root)}")
        
        # Print first few elements
        print("\nFirst few elements:")
        for i, child in enumerate(root):
            if i >= 3:  # Only show first 3 elements
                break
            print(f"  Element {i+1}: tag={child.tag}, attributes={child.attrib}")
            
    except ET.ParseError as e:
        print(f"\nError parsing {file_path}:")
        print(f"ParseError: {str(e)}")
        
        # Try to read the file contents around the error
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                line_num = int(str(e).split('line')[1].split(',')[0])
                print("\nContext around error:")
                start_line = max(0, line_num - 3)
                end_line = min(len(lines), line_num + 2)
                for i in range(start_line, end_line):
                    print(f"Line {i+1}: {lines[i].rstrip()}")
        except Exception as file_e:
            print(f"Could not read file contents: {str(file_e)}")
    except Exception as e:
        print(f"\nOther error with {file_path}:")
        print(str(e))

def main():
    data_dir = Path("TEST_CASE")
    xml_files = [
        'tls_states.xml',
        'tls_switches.xml',
        'tripinfo.xml',
        'summary.xml',
        'additional.xml',
        'collisions.xml',
        'routes.xml',
        'stops.xml',
        'tls_programs.xml',
        'tls_switch_states.xml'
    ]
    
    print("Starting XML validation...")
    for xml_file in xml_files:
        file_path = data_dir / xml_file
        if file_path.exists():
            print(f"\nChecking {xml_file}...")
            validate_xml_file(file_path)
        else:
            print(f"\n{xml_file} not found in {data_dir}")

if __name__ == "__main__":
    main()