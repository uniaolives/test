# pic_mapper.py
# Automated COBOL PIC to Rust/SQL Type Translation
# Part of the ASI-Ω Legacy Modernization Suite

import re
from typing import Dict, Optional

class CopybookParser:
    """
    Parser for COBOL Copybook clauses (PIC, REDEFINES, OCCURS).
    Transmutes hierarchical structures into Mesh-compatible metadata.
    """
    def parse_line(self, line: str) -> Optional[Dict]:
        # Regex for level, name, REDEFINES, OCCURS, and PIC
        # Example: 05 CLIENT-ID PIC 9(07).
        pattern = r"(\d+)\s+([\w-]+)(?:\s+REDEFINES\s+([\w-]+))?(?:\s+OCCURS\s+(\d+))?(?:\s+PIC\s+([9X\(\)V\d]+))?"
        match = re.search(pattern, line.strip(), re.IGNORECASE)
        if match:
            return {
                "level": int(match.group(1)),
                "name": match.group(2).replace("-", "_"),
                "redefines": match.group(3),
                "occurs": int(match.group(4)) if match.group(4) else None,
                "pic": match.group(5)
            }
        return None

class TypeMapper:
    """
    Maps COBOL types to Rust and SQL equivalents ensuring decimal precision.
    """
    @staticmethod
    def pic_to_rust(pic_clause: str) -> str:
        if not pic_clause:
            return "struct" # Nested group

        pic = pic_clause.upper()
        if 'V' in pic:
            return "Decimal" # Fixed-point decimal

        if 'X' in pic:
            return "String" # Alphanumeric

        # Numeric (9)
        size = TypeMapper.extract_size(pic)
        if size <= 4:
            return "i16"
        elif size <= 9:
            return "i32"
        elif size <= 18:
            return "i64"
        else:
            return "Decimal"

    @staticmethod
    def pic_to_sql(pic_clause: str) -> str:
        if not pic_clause:
            return "JSONB"

        pic = pic_clause.upper()
        if 'X' in pic:
            size = TypeMapper.extract_size(pic)
            return f"VARCHAR({size})"

        if 'V' in pic:
            parts = pic.split('V')
            p1 = TypeMapper.extract_size(parts[0])
            p2 = TypeMapper.extract_size(parts[1])
            precision = p1 + p2
            scale = p2
            return f"DECIMAL({precision}, {scale})"

        size = TypeMapper.extract_size(pic)
        return f"NUMERIC({size}, 0)"

    @staticmethod
    def extract_size(pic: str) -> int:
        """Extracts total length from PIC clause like 9(07)V99 -> 9."""
        count = 0
        # Find all occurrences of 9(n), X(n), 9, or X
        matches = re.findall(r'([9X])(?:\((\d+)\))?', pic)
        for char, size in matches:
            if size:
                count += int(size)
            else:
                count += 1
        return count

def main():
    parser = CopybookParser()
    mapper = TypeMapper()

    sample_lines = [
        "01 CLIENT-RECORD.",
        "  05 CLIENT-ID PIC 9(07).",
        "  05 CLIENT-NAME PIC X(30).",
        "  05 CLIENT-BALANCE PIC 9(9)V99.",
        "  05 CLIENT-PHONES OCCURS 5 PIC 9(10)."
    ]

    print("🜁 LEGACY SCHEMA TRANSMUTATION REPORT")
    print("=" * 50)

    for line in sample_lines:
        parsed = parser.parse_line(line)
        if parsed and parsed['pic']:
            rust_type = mapper.pic_to_rust(parsed['pic'])
            sql_type = mapper.pic_to_sql(parsed['pic'])
            print(f"COBOL: {parsed['name']} ({parsed['pic']})")
            print(f"  -> Rust: {rust_type}")
            print(f"  -> SQL:  {sql_type}")
            print("-" * 30)

if __name__ == "__main__":
    main()
