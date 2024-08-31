def enum_to_string(value):
    return value.value if hasattr(value, 'value') else value
