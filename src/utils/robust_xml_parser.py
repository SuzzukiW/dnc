# src/utils/robust_xml_parser.py

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustXMLParser:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.csv_dir = self.data_dir / "csv"
        self.csv_dir.mkdir(exist_ok=True)
    
    def read_file_with_debug(self, file_path: Path) -> str:
        """Read file and show debug information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Log first few lines for debugging
            logger.info(f"First few lines of {file_path.name}:")
            first_lines = content.split('\n')[:5]
            for i, line in enumerate(first_lines):
                logger.info(f"Line {i+1}: {line}")
            
            return content
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return ""

    def clean_xml_content(self, content: str, file_name: str) -> str:
        """Clean XML content before parsing."""
        logger.info(f"Cleaning {file_name}...")
        
        # Remove BOM if present
        content = content.replace('\ufeff', '')
        
        # Remove invalid comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        content = re.sub(r'-->', '', content)
        
        # Find the root element name
        root_match = re.search(r'<(\w+)[^>]*>', content)
        if not root_match:
            # If no root element found, create one based on filename
            root_name = file_name.replace('.xml', '')
            content = f'<{root_name}>\n{content}\n</{root_name}>'
            logger.info(f"Added root element <{root_name}> to {file_name}")
        
        # Fix unclosed tags
        content = re.sub(r'<([^/>]+)>([^<]*)\n', r'<\1>\2</\1>\n', content)
        
        # Fix missing quotes in attributes
        content = re.sub(r'(\w+)=(\w+)', r'\1="\2"', content)
        
        # Fix any self-closing tags
        content = re.sub(r'<([^>]+)>([^<]*)</[^>]*$', r'<\1>\2</\1>', content)
        
        # Add XML declaration if missing
        if not content.startswith('<?xml'):
            content = '<?xml version="1.0" encoding="UTF-8"?>\n' + content
        
        # Log the first few lines of cleaned content
        logger.info(f"First few lines after cleaning {file_name}:")
        first_lines = content.split('\n')[:5]
        for i, line in enumerate(first_lines):
            logger.info(f"Line {i+1}: {line}")
        
        return content

    def parse_tls_states(self) -> pd.DataFrame:
        """Parse traffic light states XML to DataFrame."""
        try:
            content = self.read_file_with_debug(self.data_dir / 'tls_states.xml')
            if not content:
                return pd.DataFrame()
            
            # Add custom wrapper if needed
            if '<tlsStates' not in content:
                content = """<?xml version="1.0" encoding="UTF-8"?>
<tlsStates>
""" + content + """
</tlsStates>"""
            
            clean_content = self.clean_xml_content(content, 'tls_states.xml')
            
            # Try to parse with ElementTree
            try:
                root = ET.fromstring(clean_content)
            except ET.ParseError as e:
                logger.error(f"XML Parse error: {str(e)}")
                logger.info("Attempting manual parsing...")
                return self.manual_parse_tls_states(clean_content)
            
            data = []
            for state in root.findall('.//tlsState'):
                row = {
                    'time': float(state.get('time', 0)),
                    'id': state.get('id', ''),
                    'programID': state.get('programID', '0'),
                    'phase': state.get('phase', '0'),
                    'state': state.get('state', '')
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(self.csv_dir / 'tls_states.csv', index=False)
            logger.info(f"Successfully parsed tls_states.xml: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing tls_states.xml: {str(e)}")
            return pd.DataFrame()
    
    def manual_parse_tls_states(self, content: str) -> pd.DataFrame:
        """Manually parse TLS states if XML parsing fails."""
        data = []
        lines = content.split('\n')
        
        for line in lines:
            if 'tlsState' in line:
                try:
                    # Extract attributes using regex
                    time_match = re.search(r'time="([^"]+)"', line)
                    id_match = re.search(r'id="([^"]+)"', line)
                    program_match = re.search(r'programID="([^"]+)"', line)
                    phase_match = re.search(r'phase="([^"]+)"', line)
                    state_match = re.search(r'state="([^"]+)"', line)
                    
                    if time_match and id_match:
                        row = {
                            'time': float(time_match.group(1)),
                            'id': id_match.group(1),
                            'programID': program_match.group(1) if program_match else '0',
                            'phase': phase_match.group(1) if phase_match else '0',
                            'state': state_match.group(1) if state_match else ''
                        }
                        data.append(row)
                except Exception as e:
                    logger.warning(f"Error parsing line: {line}\nError: {str(e)}")
                    continue
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.to_csv(self.csv_dir / 'tls_states.csv', index=False)
            logger.info(f"Successfully manually parsed tls_states.xml: {len(df)} rows")
        return df

    def process_all(self) -> bool:
        """Process all XML files to CSV."""
        logger.info("Starting XML processing...")
        success = True
        
        # Parse traffic light states
        tls_states = self.parse_tls_states()
        if tls_states.empty:
            success = False
            logger.error("Failed to parse traffic light states")
        else:
            logger.info(f"Successfully parsed {len(tls_states)} traffic light states")
            # Log some sample data
            logger.info("\nSample TLS states data:")
            logger.info(tls_states.head().to_string())
        
        return success

def main():
    parser = RobustXMLParser("TEST_CASE")
    success = parser.process_all()
    if success:
        logger.info("Successfully converted all XML files to CSV")
    else:
        logger.error("Some files could not be converted")

if __name__ == "__main__":
    main()