from __future__ import print_function, division
import regex

_reobj = None

def compile_regex():
    global _reobj

    pattern_type = \
        r'(?:' + \
            r'(?:' + \
                r'(?:' + \
                    r'(?:(?:signed|unsigned)\s+)?' + \
                    r'(?:char|byte|short|int|long)' + \
                r')' + \
                r'|ushort|uint|ulong|bool|boolean|str|string|[a-zA-Z0-9_-]+' + \
            r')' + \
            r'(?:' + \
                r'\s*' + \
                r'(?:' + \
                    r'\*|\[\s*\d*\s*\]' + \
                r')+' + \
                r'\s*' + \
                r'|\s+' + \
            r')' + \
        r')?'

    pattern_modifier = \
        r'(?:' + \
            r'(?:' + \
                r'private|protected|public|static' + \
            r')' + \
            r'(?:' + \
                r'\s+' + \
                r'(?:' + \
                    r'private|protected|public|static' + \
                r')' + \
            r')*' + \
            r'\s+' + \
        r')?'

    pattern_class = \
        r'\s*' + \
        r'(?:' + \
            r'(?P<class>' + \
                r'[a-zA-Z0-9_-]+' + \
            r')' + \
            r'\s*' + \
            r'(?:' + \
                r'::|\.' + \
            r')' + \
        r')?' + \
        r'\s*'

    pattern_name = \
        r'\s*' + \
        pattern_modifier + \
        pattern_type + \
        pattern_class + \
        r'(?P<func>' + \
            r'[a-zA-Z0-9_-]+' + \
        r')' + \
        r'\s*'

    def pattern_param(id):
        return \
            r'\s*' + \
            pattern_type + \
            r'(?P<' + id + '>' + \
                r'[a-zA-Z0-9_-]+' + \
            r')' + \
            r'\s*'

    pattern_params = \
        r'\s*' + \
        r'(?:' + \
            pattern_param('param1') + \
            r'(?:' + \
                r',' + \
                pattern_param('paramN') + \
            r')*' + \
        r')?' + \
        r'\s*'

    pattern = \
        r'^' + \
        pattern_name + \
        r'(?:' + \
            r'\(' + \
            pattern_params + \
            r'\)' \
        r')?' + \
        r'\s*' + \
        r'(?:' + \
            r'const' + \
        r')?' + \
        r'\s*;?\s*' + \
        r'$'

    _reobj = regex.compile(pattern, flags=regex.IGNORECASE)


def parse_signature(sig):
    m = _reobj.match(sig)
    if not m:
        return None

    class_name = m.group('class').strip() if m.group('class') is not None else None
    func_name = m.group('func').strip()
    param_names = []

    if m.group('param1') is not None:
        param_names.append(m.group('param1').strip())

    if m.group('paramN') is not None:
        param_names.extend([n.strip() for n in m.captures('paramN')])

    return func_name, param_names, class_name


compile_regex()
