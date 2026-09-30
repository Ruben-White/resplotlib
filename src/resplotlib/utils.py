def _combine_dicts(dict1: dict, dict2: dict, max_depth: int = 4) -> dict:
    """Recursively combine dictionaries, prioritising dictionary 2.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.
        max_depth (int): Maximum recursion depth.

    Returns:
        dict: Combined dictionary.
    """

    # Maximum depth reached -> return dictionary 2
    if max_depth == 0:
        return dict2

    # Both dictionaries are not dictionaries -> return dictionary 2
    if not isinstance(dict1, dict) and not isinstance(dict2, dict):  # noqa: SIM114
        return dict2

    # Dictionary 1 is not a dictionary -> return dictionary 2
    elif not isinstance(dict1, dict):
        return dict2

    # Dictionary 2 is not a dictionary -> return dictionary 1
    elif not isinstance(dict2, dict):
        return dict1

    # Both dictionaries are dictionaries -> combine dictionaries
    keys = list(dict1.keys()) + list(dict2.keys())
    dict_combined = {}
    for key in keys:
        # Key not in dictionary 1 -> use dictionary 2
        if key not in dict1:
            dict_combined[key] = dict2[key]

        # Key not in dictionary 2 -> use dictionary 1
        elif key not in dict2:
            dict_combined[key] = dict1[key]

        # Key in both dictionaries -> combine dictionaries
        else:
            dict_combined[key] = _combine_dicts(dict1[key], dict2[key], max_depth=max_depth - 1)

    return dict_combined


def _substitute_str_in_dict(dict1: dict, str1: str, str2: str) -> dict:
    """Recursively substitutes strings in a dictionary.

    Args:
        dict1 (dict): Dictionary to substitute strings in.
        str1 (str): String to substitute.
        str2 (str): String to substitute with.

    Returns:
        dict: Dictionary with substituted strings.
    """

    # Check if strings are None
    if str1 is None or str2 is None:
        return dict1

    # Substitute string in dictionary
    for key, value in dict1.items():
        if isinstance(value, dict):
            dict1[key] = _substitute_str_in_dict(value, str1, str2)
        elif isinstance(value, str):
            dict1[key] = value.replace(str1, str2)

    return dict1


def _substitute_inherit_str_in_dicts(dict1: dict, max_depth: int = 4) -> dict:
    """Recursively apply inheritance in a dictionary of dictionaries, substituting the "@inherits" key with the corresponding dictionary in the same parent dictionary.

    Args:
        dict1 (dict): Dictionary to apply inheritance in.
        max_depth (int): Maximum recursion depth.

    Returns:
        dict: Dictionary with applied inheritance.
    """

    # Maximum depth reached -> return dictionary
    if max_depth == 0:
        return dict1

    for key1 in dict1:  # noqa: PLC0206
        # Check if value is a dictionary
        if not isinstance(dict1[key1], dict):
            continue

        # Recursively apply inheritance in child dictionary
        dict2 = _substitute_inherit_str_in_dicts(dict1[key1], max_depth=max_depth - 1)

        # Check if "inherits" key is present
        if "inherits" not in dict2:
            continue

        # Get reference and available references
        reference = dict2.pop("inherits").lstrip("@")
        available_references = [k for k in dict1 if k != key1]

        # Apply inheritance if reference is available
        if reference in available_references:
            dict2 = _combine_dicts(dict1[reference], dict2)
            dict2 = dict(sorted(dict2.items()))
            dict1[key1] = dict2
        else:
            raise KeyError(
                f"An error occurred while processing guidelines. Inheritance reference '{reference}' in '{key1}' not found. Available references: {available_references}"
            )

    return dict1
