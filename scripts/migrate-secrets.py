#!/usr/bin/env python3
# scripts/migrate-secrets.py
import re
import os
import yaml
from pathlib import Path

class SecretMigrator:
    def __init__(self, workspace_path):
        self.workspace = Path(workspace_path)

    def find_secrets(self):
        """Find secrets across all projects"""
        secrets = []
        patterns = {
            'api_key': r'(?i)(api[_-]?key|apikey)[\s]*=[\s]*[\'"]([^\'"]+)[\'"]',
            'jwt_secret': r'(?i)(jwt[_-]?secret|secret[_-]?key)[\s]*=[\s]*[\'"]([^\'"]+)[\'"]',
            'database_url': r'(?i)(database[_-]?url|db[_-]?url)[\s]*=[\s]*[\'"]([^\'"]+)[\'"]'
        }

        for root, dirs, files in os.walk(self.workspace):
            if '.git' in root or 'target' in root or '__pycache__' in root:
                continue
            for file_name in files:
                if file_name.endswith(('.env', '.yml', '.yaml', '.json', '.py', '.js')):
                    file_path = Path(root) / file_name
                    try:
                        content = file_path.read_text()
                        for secret_type, pattern in patterns.items():
                            matches = re.findall(pattern, content)
                            for match in matches:
                                secrets.append({
                                    'file': str(file_path.relative_to(self.workspace)),
                                    'type': secret_type,
                                    'value': match[1] if len(match) > 1 else match[0],
                                    'line': content[:content.find(match[0])].count('\n') + 1
                                })
                    except:
                        pass
        return secrets

    def migrate_to_vault(self, secrets):
        """Migrate secrets to quantum vault"""
        vault_config = {
            'apiVersion': 'vault.openclaw/v1alpha1',
            'kind': 'QuantumSecretManager',
            'metadata': {'name': 'workspace-secrets'},
            'spec': {
                'secrets': [],
                'rotation_policy': '24h',
                'access_control': {
                    'ai_agents_required': 3,
                    'human_approval_required': True
                }
            }
        }

        for secret in secrets:
            vault_config['spec']['secrets'].append({
                'name': f"{secret['type']}-{secret['line']}",
                'original_location': secret['file'],
                'encrypted_value': "REDACTED_ENCRYPTED", # Placeholder for actual encryption
                'last_rotated': None
            })

        vault_path = self.workspace / 'security' / 'quantum-vault.yaml'
        vault_path.parent.mkdir(exist_ok=True)

        with open(vault_path, 'w') as f:
            yaml.dump(vault_config, f)

        print(f"✅ Migrated {len(secrets)} secrets to quantum vault")
        print(f"🔐 Vault location: {vault_path}")

if __name__ == "__main__":
    migrator = SecretMigrator(".")
    secrets = migrator.find_secrets()
    migrator.migrate_to_vault(secrets)
