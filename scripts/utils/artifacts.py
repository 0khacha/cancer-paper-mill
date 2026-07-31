import re
import html
import unicodedata

# =====================================================================
# Artifact detection regex
# =====================================================================
headers_list = [
    "MATERIALS AND METHODS", "PATIENTS AND METHODS", "BACKGROUND",
    "OBJECTIVE", "OBJECTIVES", "AIM", "AIMS", "INTRODUCTION",
    "PURPOSE", "METHOD", "METHODS", "RESULT", "RESULTS",
    "CONCLUSION", "CONCLUSIONS", "DISCUSSION", "SIGNIFICANCE", "DESIGN"
]
headers_pat = "|".join(headers_list)
header_regex = re.compile(
    rf'([.,!;?\-\s\(\[\]\)]*)\b({headers_pat})\b\s*[\]\)]*\s*:\s*([\(\[\]\)\-\s,;:]*)',
    re.IGNORECASE
)

def detect_glued_header(text):
    """
    Frozen glued-header detection.
    Returns True if any glued header is found.
    """
    if not text:
        return False
    
    matches = list(header_regex.finditer(text))
    rebuilt_so_far = ""
    
    for m in matches:
        pre = m.group(1)
        start_idx = m.start()
        before_match = text[len(rebuilt_so_far):start_idx]
        rebuilt_so_far += before_match + m.group(0)
        
        is_start = (before_match.strip() == "" and rebuilt_so_far.replace(m.group(0), "").strip() == "")
        
        if not is_start:
            pre_stripped = pre.strip()
            if not pre_stripped or pre_stripped[-1] not in ['.', '!', '?']:
                return True
            elif len(pre) > 0 and pre[0] in ['.', '!', '?'] and not pre.endswith(' '):
                return True
    
    return False

def analyze_all_artifacts(text):
    """Full artifact analysis."""
    if not text:
        return {'glued_header': False, 'stray_punctuation': False, 
                'whitespace_irregularity': False, 'unicode_entity': False}
    
    result = {'glued_header': False, 'stray_punctuation': False,
              'whitespace_irregularity': False, 'unicode_entity': False}
    
    # Unicode/entity checks
    if unicodedata.normalize('NFC', text) != text:
        result['unicode_entity'] = True
    if re.search(r'[\u00a0\u200b\u202f\u2007\xa0]', text):
        result['unicode_entity'] = True
    if html.unescape(text) != text:
        result['unicode_entity'] = True
    
    # Header artifacts
    matches = list(header_regex.finditer(text))
    rebuilt_so_far = ""
    
    for m in matches:
        pre = m.group(1)
        post = m.group(3)
        start_idx = m.start()
        end_idx = m.end()
        before_match = text[len(rebuilt_so_far):start_idx]
        rebuilt_so_far += before_match + m.group(0)
        
        is_start = (before_match.strip() == "" and rebuilt_so_far.replace(m.group(0), "").strip() == "")
        
        if not is_start:
            pre_stripped = pre.strip()
            if not pre_stripped or pre_stripped[-1] not in ['.', '!', '?']:
                result['glued_header'] = True
            elif len(pre) > 0 and pre[0] in ['.', '!', '?'] and not pre.endswith(' '):
                result['glued_header'] = True
            
            if any(c in pre for c in '([)]-,-;:'):
                result['stray_punctuation'] = True
        else:
            if pre.strip():
                result['stray_punctuation'] = True
        
        if any(c in post for c in '([)]-,-;:'):
            result['stray_punctuation'] = True
        
        if len(re.findall(r'\s{2,}', pre)) > 0 or len(re.findall(r'\s{2,}', post)) > 0:
            result['whitespace_irregularity'] = True
        if post == "" and end_idx < len(text):
            result['whitespace_irregularity'] = True
            
    return result
