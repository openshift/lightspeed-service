"""Integration tests for config loader."""

import copy
import itertools
import random

import pytest
import yaml

from ols import config
from ols.constants import DEFAULT_CONFIGURATION_FILE
from ols.utils.checks import InvalidConfigurationError
from tests.integration.random_payload_generator import RandomPayloadGenerator

CORRECT_CONFIG_FILE = "tests/config/config_for_integration_tests.yaml"
MINIMAL_CONFIG_FILE = "tests/config/minimal_config.yaml"


def test_load_proper_config():
    """Test loading proper config."""
    # proper config should be loaded w/o throwing any exception
    config.reload_from_yaml_file(CORRECT_CONFIG_FILE)


def test_load_non_existent_config():
    """Test how loading of non-existent config is handled."""
    with pytest.raises(
        FileNotFoundError, match=r"tests/config/non_existent_config.yaml"
    ):
        config.reload_from_yaml_file("tests/config/non_existent_config.yaml")


def test_load_improper_config():
    """Test how loading of improper config is handled."""
    with pytest.raises(InvalidConfigurationError, match="invalid provider type"):
        config.reload_from_yaml_file("tests/config/invalid_config.yaml")


def load_config_file():
    """Read proper configuration file and deserialize from YAML format."""
    with open(MINIMAL_CONFIG_FILE, encoding="utf-8") as fin:
        return yaml.safe_load(fin)


def write_config_file(cfg_filename, cfg):
    """Serialize configuration into YAML format and write it to file."""
    with open(cfg_filename, "w", encoding="utf-8") as fout:
        yaml.dump(cfg, fout, default_flow_style=False)


def remove_items_one_iteration(
    original_payload, items_count, remove_flags, selector=None
):
    """One iteration of algorithm to remove item or items from the original payload."""
    if selector is None:
        keys = list(original_payload.keys())
    else:
        keys = list(original_payload[selector].keys())

    # perform deep copy of original payload before modification
    new_payload = copy.deepcopy(original_payload)

    removed_keys = []
    for i in range(items_count):
        # should be the item under index i removed?
        remove_flag = remove_flags[i]
        if remove_flag:
            key = keys[i]
            if selector is None:
                del new_payload[key]
            else:
                del new_payload[selector][key]
            removed_keys.append(key)
    return new_payload


def remove_items(original_payload, selector=None):
    """Algorithm to remove items from original payload."""
    if selector is None:
        items_count = len(original_payload)
    else:
        items_count = len(original_payload[selector])

    # lexicographics ordering to make multiple test runs stable
    remove_flags_list = list(itertools.product([True, False], repeat=items_count))

    # the last item contains (False, False, False...) and we are not interested
    # in removing ZERO items
    remove_flags_list = remove_flags_list[:-1]

    # construct new payload with some item(s) removed
    new_payloads = [
        remove_items_one_iteration(
            original_payload, items_count, remove_flags, selector
        )
        for remove_flags in remove_flags_list
    ]
    return new_payloads


def test_load_config_with_removed_items(tmpdir, subtests):
    """Test how broken config with missing items is handled."""
    cfg = load_config_file()
    broken_configs = remove_items(cfg)

    for i, broken_config in enumerate(broken_configs):
        with subtests.test(msg=f"removed_item_{i}", i=i):
            cfg_filename = tmpdir + "/" + DEFAULT_CONFIGURATION_FILE
            write_config_file(cfg_filename, broken_config)

            with pytest.raises(InvalidConfigurationError):
                config.reload_from_yaml_file(cfg_filename)


def test_load_config_with_removed_items_from_ols_config_section(tmpdir, subtests):
    """Test how broken config with missing items in ols_config section is handled."""
    cfg = load_config_file()
    broken_configs = remove_items(cfg, "ols_config")

    for i, broken_config in enumerate(broken_configs):
        with subtests.test(msg=f"removed_item_{i}", i=i):
            cfg_filename = tmpdir + "/" + DEFAULT_CONFIGURATION_FILE
            write_config_file(cfg_filename, broken_config)

            with pytest.raises(InvalidConfigurationError):
                config.reload_from_yaml_file(cfg_filename)


def mutate_items_one_iteration(original_payload, how_many):
    """One iteration of algorithm to mutate items in original payload."""
    # Perform deep copy of original payload.
    new_payload = copy.deepcopy(original_payload)
    rpg = RandomPayloadGenerator()

    for _ in range(how_many):
        selected_key = random.choice(list(original_payload.keys()))  # noqa: S311
        new_value = rpg.generate_random_payload()
        new_payload[selected_key] = new_value

    return new_payload


def mutate_items(original_payload, how_many):
    """Algorithm to mutate items with random values in original payload."""
    new_payloads = [
        mutate_items_one_iteration(original_payload, i) for i in range(how_many)
    ]
    return new_payloads


def test_load_config_with_mutated_items(tmpdir, subtests):
    """Test how broken config is handled."""
    cfg = load_config_file()
    broken_configs = mutate_items(cfg, 10)

    for i, broken_config in enumerate(broken_configs):
        with subtests.test(msg=f"mutated_item_{i}", i=i):
            cfg_filename = tmpdir + "/" + DEFAULT_CONFIGURATION_FILE
            write_config_file(cfg_filename, broken_config)

            with pytest.raises(Exception):
                config.reload_from_yaml_file(cfg_filename)
