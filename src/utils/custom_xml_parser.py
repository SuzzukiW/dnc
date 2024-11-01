# src/utils/custom_xml_parser.py

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomXMLParser:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.csv_dir = self.data_dir / "csv"
        self.csv_dir.mkdir(exist_ok=True)
    
    def clean_xml_content(self, content: str) -> str:
        """Clean XML content before parsing."""
        # Remove invalid XML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        content = re.sub(r'-->', '', content)
        
        # Fix unclosed tags
        content = re.sub(r'<([^>]*)>([^<]*)</[^>]*$', r'<\1>\2</\1>', content)
        
        # Fix broken attributes
        content = re.sub(r'(\w+)=(\w+)', r'\1="\2"', content)
        
        return content

    def parse_tls_states(self) -> pd.DataFrame:
        """Parse traffic light states XML to DataFrame."""
        try:
            with open(self.data_dir / 'tls_states.xml', 'r') as f:
                content = f.read()
            
            clean_content = self.clean_xml_content(content)
            root = ET.fromstring(clean_content)
            
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
    
    def parse_summary(self) -> pd.DataFrame:
        """Parse summary XML to DataFrame."""
        try:
            with open(self.data_dir / 'summary.xml', 'r') as f:
                content = f.read()
            
            clean_content = self.clean_xml_content(content)
            root = ET.fromstring(clean_content)
            
            data = []
            for step in root.findall('.//step'):
                row = {
                    'time': float(step.get('time', 0)),
                    'loaded': int(step.get('loaded', 0)),
                    'inserted': int(step.get('inserted', 0)),
                    'running': int(step.get('running', 0)),
                    'waiting': int(step.get('waiting', 0)),
                    'ended': int(step.get('ended', 0)),
                    'meanWaitingTime': float(step.get('meanWaitingTime', 0)),
                    'meanSpeed': float(step.get('meanSpeed', 0))
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(self.csv_dir / 'summary.csv', index=False)
            logger.info(f"Successfully parsed summary.xml: {len(df)} rows")
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
            root = ET.fromstring(clean_content)
            
            data = []
            for trip in root.findall('.//tripinfo'):
                row = {
                    'id': trip.get('id', ''),
                    'depart': float(trip.get('depart', 0)),
                    'departLane': trip.get('departLane', ''),
                    'departSpeed': float(trip.get('departSpeed', 0)),
                    'duration': float(trip.get('duration', 0)),
                    'waitingTime': float(trip.get('waitingTime', 0)),
                    'timeLoss': float(trip.get('timeLoss', 0))
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(self.csv_dir / 'tripinfo.csv', index=False)
            logger.info(f"Successfully parsed tripinfo.xml: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing tripinfo.xml: {str(e)}")
            return pd.DataFrame()

    def process_all(self) -> bool:
        """Process all XML files to CSV."""
        success = True
        
        # Parse each file
        tls_states = self.parse_tls_states()
        summary = self.parse_summary()
        tripinfo = self.parse_tripinfo()
        
        # Check if we have the minimum required data
        if tls_states.empty:
            logger.error("Could not parse traffic light states data")
            success = False
        if summary.empty:
            logger.error("Could not parse summary data")
            success = False
            
        return success

def main():
    parser = CustomXMLParser("TEST_CASE")
    success = parser.process_all()
    if success:
        logger.info("Successfully converted all XML files to CSV")
    else:
        logger.error("Some files could not be converted")

if __name__ == "__main__":
    main()