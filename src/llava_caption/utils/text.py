import re

def preprocess_text(text: str) -> str:
    """Clean and standardize input text."""
    replacements = {
        r'(photo of\s\w+),': 'photo of',
        'Cybergirl': 'woman',
        'Cyberpunk man': 'man',
        'photograph': '',
        'film': '',
        'BREAK': '',
        'professional': '',
        'highly detailed': ''
    }
    
    result = text
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result)
        
    return result.strip()