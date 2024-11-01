# src/utils/sumo_xml_parser.py

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SUMOXMLParser:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.csv_dir = self.data_dir / "csv"
        self.csv_dir.mkdir(exist_ok=True)
    
    def clean_xml_content(self, content: str) -> str:
        """Clean SUMO XML content before parsing."""
        # Remove XML declaration duplicates
        content = re.sub(r'<\?xml.*?\?>', '', content, flags=re.MULTILINE)
        content = '<?xml version="1.0" encoding="UTF-8"?>\n' + content
        
        # Remove comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # Remove configuration section
        content = re.sub(r'<sumoConfiguration.*?</sumoConfiguration>', '', content, flags=re.DOTALL)
        
        return content.strip()

    def extract_attributes(self, line: str, patterns: dict) -> dict:
        """Extract attributes from a line using patterns."""
        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                try:
                    value = match.group(1)
                    # Convert numeric values
                    if key in ['time', 'waiting', 'meanWaitingTime', 'meanSpeed']:
                        value = float(value)
                    elif key in ['running', 'inserted', 'ended']:
                        value = int(value)
                    result[key] = value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error converting value for {key}: {str(e)}")
                    result[key] = 0
        return result

    def parse_tls_states(self) -> pd.DataFrame:
        """Parse traffic light states XML to DataFrame."""
        try:
            with open(self.data_dir / 'tls_states.xml', 'r') as f:
                content = f.read()
            
            clean_content = self.clean_xml_content(content)
            
            patterns = {
                'time': r'time="([^"]+)"',
                'id': r'id="([^"]+)"',
                'programID': r'programID="([^"]+)"',
                'phase': r'phase="([^"]+)"',
                'state': r'state="([^"]+)"'
            }
            
            data = []
            for line in clean_content.split('\n'):
                if 'tlsState' in line:
                    row = self.extract_attributes(line, patterns)
                    if row:
                        data.append(row)
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.to_csv(self.csv_dir / 'tls_states.csv', index=False)
                logger.info(f"Successfully parsed {len(df)} traffic light states")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing tls_states.xml: {str(e)}")
            return pd.DataFrame()

    def parse_summary(self) -> pd.DataFrame:
        """Parse summary XML to DataFrame."""
        try:
            with open(self.data_dir / 'summary.xml', 'r') as f:
                content = f.read()
            
            clean_content = self.clean_xml_content(content)
            
            patterns = {
                'time': r'time="([^"]+)"',
                'running': r'running="([^"]+)"',
                'waiting': r'waiting="([^"]+)"',
                'meanWaitingTime': r'meanWaitingTime="([^"]+)"',
                'meanSpeed': r'meanSpeed="([^"]+)"',
                'ended': r'ended="([^"]+)"',
                'inserted': r'inserted="([^"]+)"'
            }
            
            data = []
            for line in clean_content.split('\n'):
                if 'step' in line:
                    row = self.extract_attributes(line, patterns)
                    if row:
                        data.append(row)
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.to_csv(self.csv_dir / 'summary.csv', index=False)
                logger.info(f"Successfully parsed {len(df)} summary records")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing summary.xml: {str(e)}")
            return pd.DataFrame()
    
    def parse_tripinfo(self) -> pd.DataFrame:
        """Parse trip information XML to DataFrame."""
        try:
            with open(self.data_dir / 'tripinfo.xml', 'r') as f:
                content = f.read()
            
            clean_content = self.clean_xml_content(content)
            
            patterns = {
                'id': r'id="([^"]+)"',
                'depart': r'depart="([^"]+)"',
                'departLane': r'departLane="([^"]+)"',
                'departSpeed': r'departSpeed="([^"]+)"',
                'departDelay': r'departDelay="([^"]+)"',
                'arrival': r'arrival="([^"]+)"',
                'duration': r'duration="([^"]+)"',
                'waitingTime': r'waitingTime="([^"]+)"',
                'timeLoss': r'timeLoss="([^"]+)"'
            }
            
            data = []
            for line in clean_content.split('\n'):
                if 'tripinfo' in line:
                    row = self.extract_attributes(line, patterns)
                    if row:
                        data.append(row)
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.to_csv(self.csv_dir / 'tripinfo.csv', index=False)
                logger.info(f"Successfully parsed {len(df)} trip records")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing tripinfo.xml: {str(e)}")
            return pd.DataFrame()

    def process_all(self) -> bool:
        """Process all XML files to CSV."""
        logger.info("Starting SUMO XML processing...")
        
        # Parse all files
        tls_states = self.parse_tls_states()
        summary = self.parse_summary()
        tripinfo = self.parse_tripinfo()
        
        # Check results
        if tls_states.empty:
            logger.error("Failed to parse traffic light states")
            return False
        
        if summary.empty:
            logger.error("Failed to parse summary data")
            return False
        
        # Log sample data
        logger.info("\nSample TLS states data:")
        logger.info(tls_states.head().to_string())
        
        logger.info("\nSample summary data:")
        logger.info(summary.head().to_string() if not summary.empty else "No summary data")
        
        return True

def main():
    parser = SUMOXMLParser("TEST_CASE")
    success = parser.process_all()
    if success:
        logger.info("Successfully converted all SUMO XML files to CSV")
    else:
        logger.error("Some files could not be converted")

if __name__ == "__main__":
    main()