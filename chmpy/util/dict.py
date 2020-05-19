import collections


def recursive_dict_update(dict_to, dict_from):
    """Iterate through a dict, updating items recursively from a second
    dictionary
    >>> d1 = {'test': {'test_val': 3}}
    >>> d2 = {'test': {'test_val': 5}, 'other': 3}
    >>> recursive_dict_update(d1, d2)
    {'test': {'test_val': 5}, 'other': 3}
    """
    for key, val in dict_from.items():
        if isinstance(val, collections.Mapping):
            dict_to[key] = recursive_dict_update(dict_to.get(key, {}), val)
        else:
            dict_to[key] = val
    return dict_to


def nested_dict_delete(root, key, sep="."):
    """Iterate through a dict, deleting items
    recursively based on a key
    >>> d1 = {'test': {'test_val': 3}}
    >>> d2 = {'test': {'test_val': 5, 'test_val_2': 7}, 'other': 3}
    >>> nested_dict_delete(d1, 'test.test_val')
    >>> d1
    {}
    >>> nested_dict_delete(d2, 'test.test_val')
    >>> d2
    {'test': {'test_val_2': 7}, 'other': 3}
    """

    levels = key.split(sep)
    level_key = levels[0]
    if level_key in root:
        if isinstance(root[level_key], collections.MutableMapping):
            nested_dict_delete(root[level_key], sep.join(levels[1:]), sep=sep)
            if not root[level_key]:
                del root[level_key]
        else:
            del root[level_key]
    else:
        raise KeyError
