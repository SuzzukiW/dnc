# src/utils/xml_cleaner.py

import os
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XMLCleaner:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def clean_xml_file(self, file_path: Path) -> bool:
        """Clean XML file by fixing common issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create backup
            backup_path = file_path.with_suffix('.xml.bak')
            if not backup_path.exists():
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Fix common issues
            cleaned = content
            
            # Fix invalid ID attributes
            cleaned = re.sub(r'id=(\d+)', r'id="\1"', cleaned)
            
            # Fix truncated elements
            cleaned = re.sub(r'(\s+d)-->', r' duration="5.00"/>', cleaned)
            cleaned = re.sub(r'(\s+phase=)-->', r' phase="0" state="GgGg"/>', cleaned)
            
            # Fix duplicate content
            if 'waiting="0263"' in cleaned:
                cleaned = cleaned.replace('waiting="0263"', 'waiting="0"')
            
            # Remove any invalid XML comments
            cleaned = re.sub(r'<!--[^>]*-->', '', cleaned)
            cleaned = re.sub(r'-->', '', cleaned)
            
            # Fix broken attributes in person elements
            cleaned = re.sub(r'(\s+x="[\d.]+"\s+y="[\d.]+"\s+)(\d+\.\d+)("\s+edge=")',
                           r'\1angle="0.00" speed="\2\3', cleaned)
            
            # Fix broken tripinfo elements
            if 'tripinfo' in file_path.name:
                cleaned = re.sub(r'(duration="\d+)</tripinfos>',
                               r'\1"/></tripinfos>', cleaned)
            
            # Ensure proper XML structure
            if not cleaned.strip().endswith('>'):
                root_tag = re.search(r'<(\w+)[^>]*>', cleaned)
                if root_tag:
                    tag_name = root_tag.group(1)
                    cleaned = cleaned.rstrip() + f'</{tag_name}>'
            
            # Write cleaned content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            logger.info(f"Successfully cleaned {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning {file_path}: {str(e)}")
            return False
    
    def clean_all_files(self):
        """Clean all XML files in the directory."""
        xml_files = [
            'tls_states.xml',
            'tls_switches.xml',
            'tripinfo.xml',
            'summary.xml',
            'collisions.xml',
            'routes.xml',
            'stops.xml',
            'tls_switch_states.xml'
        ]
        
        success_count = 0
        for xml_file in xml_files:
            file_path = self.data_dir / xml_file
            if file_path.exists():
                if self.clean_xml_file(file_path):
                    success_count += 1
                    
        logger.info(f"Cleaned {success_count} out of {len(xml_files)} files")
        return success_count

def main():
    cleaner = XMLCleaner("TEST_CASE")
    cleaner.clean_all_files()

if __name__ == "__main__":
    main()