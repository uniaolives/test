# arkhe/schema_validator.py
import json
import re
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import logging

try:
    from pydantic import BaseModel, ValidationError, create_model
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

@dataclass
class ValidationResult:
    is_valid: bool
    data: Optional[Any] = None
    errors: List[str] = None
    raw_content: str = ""
    retry_recommended: bool = False

class SchemaValidator:
    """
    Validação de schemas para LLM outputs.
    Garante que x (output) respeita a estrutura esperada (+1).
    """

    def __init__(self, schema: Optional[Dict] = None):
        self.schema = schema
        self.logger = logging.getLogger("arkhe.validator")
        self.validation_history: List[ValidationResult] = []

        if PYDANTIC_AVAILABLE and schema:
            self.pydantic_model = self._create_pydantic_model(schema)
        else:
            self.pydantic_model = None

    def _create_pydantic_model(self, schema: Dict):
        """Cria modelo Pydantic dinâmico a partir de schema JSON."""
        fields = {}
        for field_name, field_info in schema.get("properties", {}).items():
            field_type = self._json_type_to_python(field_info.get("type", "string"))
            fields[field_name] = (field_type, ...)

        return create_model("DynamicSchema", **fields)

    def _json_type_to_python(self, json_type: str):
        mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        return mapping.get(json_type, str)

    def validate(self, content: str, attempt_recovery: bool = True) -> ValidationResult:
        """
        Valida conteúdo contra schema.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            if attempt_recovery:
                recovered = self._attempt_json_recovery(content)
                if recovered:
                    return self.validate(recovered, attempt_recovery=False)

            result = ValidationResult(
                is_valid=False,
                errors=[f"JSON parse error: {str(e)}"],
                raw_content=content,
                retry_recommended=True
            )
            self.validation_history.append(result)
            return result

        if self.pydantic_model:
            try:
                validated = self.pydantic_model(**data)
                result = ValidationResult(
                    is_valid=True,
                    data=validated.model_dump() if hasattr(validated, 'model_dump') else data,
                    raw_content=content
                )
            except ValidationError as e:
                errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                result = ValidationResult(
                    is_valid=False,
                    errors=errors,
                    raw_content=content,
                    retry_recommended=True
                )
        else:
            result = ValidationResult(
                is_valid=True,
                data=data,
                raw_content=content
            )

        self.validation_history.append(result)
        return result

    def _attempt_json_recovery(self, content: str) -> Optional[str]:
        """Tenta recuperar JSON malformado."""
        strategies = [
            self._extract_json_from_markdown,
            self._fix_trailing_commas,
            self._complete_truncated_json,
            self._escape_invalid_chars,
        ]

        for strategy in strategies:
            try:
                recovered = strategy(content)
                if recovered:
                    json.loads(recovered)
                    self.logger.info(f"JSON recovered using {strategy.__name__}")
                    return recovered
            except Exception:
                continue
        return None

    def _extract_json_from_markdown(self, content: str) -> Optional[str]:
        patterns = [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'(\{.*\})']
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1)
        return None

    def _fix_trailing_commas(self, content: str) -> str:
        return re.sub(r',(\s*[}\]])', r'\1', content)

    def _complete_truncated_json(self, content: str) -> str:
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        completed = content
        completed += ']' * max(0, open_brackets)
        completed += '}' * max(0, open_braces)
        return completed

    def _escape_invalid_chars(self, content: str) -> str:
        return content.replace('\n', '\\n').replace('\t', '\\t')

    def get_validation_stats(self) -> Dict[str, Any]:
        if not self.validation_history:
            return {}
        total = len(self.validation_history)
        valid = sum(1 for r in self.validation_history if r.is_valid)
        return {
            "total_validations": total,
            "valid_count": valid,
            "invalid_count": total - valid,
            "success_rate": valid / total
        }
