# scripts/rfc_submission.py - Automação de submissão RFC

import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import os

class RFCSubmission:
    """Gerencia submissão de RFC para IETF"""

    def __init__(self):
        self.datatracker_url = "https://datatracker.ietf.org"
        self.xml_template = "templates/rfc-xml-template.xml"

    def generate_xml(self, draft_text: str, metadata: dict) -> str:
        """Gera XML no formato RFC v3"""

        root = ET.Element("rfc")
        root.set("docName", f"draft-torino-qhttp-{metadata['version']}")
        root.set("category", "std")
        root.set("ipr", "trust200902")

        # Front matter
        front = ET.SubElement(root, "front")
        title = ET.SubElement(front, "title")
        title.text = "The Quantum Hypertext Transfer Protocol (qhttp)"

        # Authors
        for author in metadata['authors']:
            author_elem = ET.SubElement(front, "author")
            author_elem.set("fullname", author['name'])
            author_elem.set("role", author.get('role', ''))

            org = ET.SubElement(author_elem, "organization")
            org.text = author['org']

            email = ET.SubElement(author_elem, "email")
            email.text = author['email']

        # Date
        date = ET.SubElement(front, "date")
        date.set("year", str(datetime.now().year))
        date.set("month", datetime.now().strftime("%B"))

        # Abstract
        abstract = ET.SubElement(front, "abstract")
        t = ET.SubElement(abstract, "t")
        t.text = metadata['abstract']

        # Middle (conteúdo)
        middle = ET.SubElement(root, "middle")

        # Converte markdown para XML
        sections = self._parse_markdown_sections(draft_text)
        for section in sections:
            section_elem = ET.SubElement(middle, "section")
            section_elem.set("title", section['title'])
            section_elem.set("anchor", section['anchor'])

            for para in section['paragraphs']:
                t = ET.SubElement(section_elem, "t")
                t.text = para

        # Back matter
        back = ET.SubElement(root, "back")

        # Referências
        references = ET.SubElement(back, "references")
        references.set("title", "Normative References")

        return ET.tostring(root, encoding='unicode')

    def _parse_markdown_sections(self, text: str):
        """Simple markdown parser for RFC sections"""
        sections = []
        current_section = None

        for line in text.split('\n'):
            if line.startswith('# '):
                if current_section:
                    sections.append(current_section)
                title = line[2:].strip()
                current_section = {
                    'title': title,
                    'anchor': title.lower().replace(' ', '-'),
                    'paragraphs': []
                }
            elif line.strip() and current_section:
                current_section['paragraphs'].append(line.strip())

        if current_section:
            sections.append(current_section)
        return sections

    def submit_to_datatracker(self, xml_content: str,
                             submission_password: str) -> dict:
        """Submete para IETF DataTracker"""

        url = f"{self.datatracker_url}/api/submit"

        files = {
            'xml': ('draft.xml', xml_content, 'application/xml'),
        }

        data = {
            'name': 'draft-torino-qhttp-00',
            'rev': '00',
            'group': 'httpbis',  # HTTP Working Group
            'submission_password': submission_password,
        }

        try:
            response = requests.post(url, files=files, data=data)
            return response.json()
        except:
            return {'status': 'error'}

    def validate_with_idnits(self, xml_path: str) -> dict:
        """Valida com idnits (ferramenta IETF)"""

        try:
            result = subprocess.run(
                ["idnits", "--verbose", xml_path],
                capture_output=True,
                text=True
            )

            return {
                'valid': result.returncode == 0,
                'errors': self._parse_idnits_errors(result.stdout),
                'warnings': self._parse_idnits_warnings(result.stdout),
                'output': result.stdout
            }
        except:
            return {'valid': False, 'output': 'idnits not installed'}

    def _parse_idnits_errors(self, output):
        errors = []
        for line in output.split('\n'):
            if 'error' in line.lower():
                errors.append(line.strip())
        return errors

    def _parse_idnits_warnings(self, output):
        warnings = []
        for line in output.split('\n'):
            if 'warning' in line.lower():
                warnings.append(line.strip())
        return warnings

    def announce_to_lists(self, announcement: str):
        """Anuncia nas listas relevantes"""

        lists = [
            'httpbis@ietf.org',
            'quantum-internet@irtf.org',
            'asi-safety@merkabah-cy.org'
        ]

        for lst in lists:
            self._send_email(
                to=lst,
                subject="[ANNOUNCE] New Draft: The qhttp:// Protocol",
                body=announcement
            )

    def _send_email(self, to, subject, body): pass

if __name__ == "__main__":
    # Executa submissão
    rfc = RFCSubmission()

    rfc_path = "RFC_QHTTP.md"
    if os.path.exists(rfc_path):
        with open(rfc_path) as f:
            draft = f.read()

        metadata = {
            'version': '00',
            'authors': [
                {'name': 'Marco Torino', 'org': 'Merkabah Research',
                 'email': 'm.torino@merkabah-cy.org'},
                {'name': 'Dr. Sarah Chen', 'org': 'MIT CSAIL',
                 'email': 's.chen@mit.edu'}
            ],
            'abstract': 'This document defines the Quantum Hypertext Transfer Protocol...'
        }

        xml = rfc.generate_xml(draft, metadata)

        # Valida
        validation = rfc.validate_with_idnits("draft.xml")
        print(f"Validation: {validation.get('valid')}")

        if validation.get('valid'):
            # Submete
            password = os.environ.get('IETF_PASSWORD', '')
            if password:
                result = rfc.submit_to_datatracker(xml, submission_password=password)
                print(f"Submission: {result}")
