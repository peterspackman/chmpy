"""
Parser for RASPA3 output files.

This module provides functions to parse both text and JSON outputs from RASPA3
simulations and extract key information such as adsorption data, energy values,
and more.
"""

import json
import re
from pathlib import Path
from typing import Any


def parse_raspa_json(file_path: str | Path) -> dict[str, Any]:
    """
    Parse a RASPA3 JSON output file.

    Args:
        file_path: Path to the JSON output file.

    Returns:
        Dictionary containing the parsed data.
    """
    file_path = Path(file_path)

    with open(file_path) as f:
        data = json.load(f)

    return data


def parse_adsorption_data(content: str) -> dict[str, dict[str, dict[str, float]]]:
    """
    Parse adsorption data from RASPA output content.

    Args:
        content: Text content of the RASPA output file.

    Returns:
        Dictionary with adsorption data for each component.
    """
    adsorption_data = {}

    # Find the last cycle section which contains the final results
    cycles = list(re.finditer(r"Current cycle: (\d+) out of \d+", content))
    if not cycles:
        return adsorption_data

    last_cycle_match = cycles[-1]
    last_cycle_pos = last_cycle_match.start()
    last_cycle_data = content[last_cycle_pos:]

    # Extract component names
    component_matches = re.findall(r"Component\s+\d+\s+\[(.*?)\]", last_cycle_data)
    for component in component_matches:
        component = component.strip()
        adsorption_data[component] = {"absolute": {}, "excess": {}}

    # Find adsorption data sections
    # Pattern matches "absolute adsorption:" followed by data, then "excess adsorption:" and data
    adsorption_pattern = (
        r"absolute adsorption:\s+(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+molecules.*?"
        + r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+mol/kg.*?"
        + r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+mg/g.*?"
        + r"excess adsorption:\s+(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+molecules.*?"
        + r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+mol/kg.*?"
        + r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+mg/g"
    )

    adsorption_matches = re.findall(adsorption_pattern, last_cycle_data, re.DOTALL)

    if adsorption_matches and component_matches:
        for i, component in enumerate(component_matches):
            if i < len(adsorption_matches):
                component = component.strip()
                abs_mol, abs_mol_kg, abs_mg_g, exc_mol, exc_mol_kg, exc_mg_g = (
                    adsorption_matches[i]
                )

                adsorption_data[component]["absolute"] = {
                    "molecules": float(abs_mol),
                    "mol_per_kg": float(abs_mol_kg),
                    "mg_per_g": float(abs_mg_g),
                }

                adsorption_data[component]["excess"] = {
                    "molecules": float(exc_mol),
                    "mol_per_kg": float(exc_mol_kg),
                    "mg_per_g": float(exc_mg_g),
                }

    return adsorption_data


def parse_energy_data(content: str) -> dict[str, float]:
    """
    Parse energy data from RASPA output content.

    Args:
        content: Text content of the RASPA output file.

    Returns:
        Dictionary with energy components.
    """
    energy_data = {}

    # Find the last cycle section which contains the final results
    cycles = list(re.finditer(r"Current cycle: (\d+) out of \d+", content))
    if not cycles:
        return energy_data

    last_cycle_match = cycles[-1]
    last_cycle_pos = last_cycle_match.start()
    last_cycle_data = content[last_cycle_pos:]

    # Extract total energy
    total_energy_match = re.search(
        r"Total potential energy/kʙ\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\((.*?)\)\s+\[K\]",
        last_cycle_data,
        re.DOTALL,
    )

    if total_energy_match:
        energy_data["total_potential"] = float(total_energy_match.group(1))

        # Extract energy components
        energy_components = re.findall(
            r"([\w\s\(\)-]+)/kʙ\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\((.*?)\)\s+\[K\]",
            last_cycle_data,
        )

        for comp_name, value, _avg_value in energy_components:
            comp_name = comp_name.strip()
            if comp_name != "Total potential energy":  # Already captured
                energy_data[comp_name.lower().replace(" ", "_")] = float(value)

    return energy_data


def parse_pressure_data(content: str) -> dict[str, float]:
    """
    Parse pressure data from RASPA output content.

    Args:
        content: Text content of the RASPA output file.

    Returns:
        Dictionary with pressure data.
    """
    pressure_data = {}

    # Find the last cycle section which contains the final results
    cycles = list(re.finditer(r"Current cycle: (\d+) out of \d+", content))
    if not cycles:
        return pressure_data

    last_cycle_match = cycles[-1]
    last_cycle_pos = last_cycle_match.start()
    last_cycle_data = content[last_cycle_pos:]

    # Extract pressure data
    pressure_match = re.search(
        r"Ideal-gas pressure:\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\+/\s+.*?\s+\[bar\]\s+"
        r"Excess pressure:\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\+/\s+.*?\s+\[bar\]\s+"
        r"Pressure:\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\+/\s+.*?\s+\[bar\]",
        last_cycle_data,
        re.DOTALL,
    )

    if pressure_match:
        pressure_data["ideal_gas"] = float(pressure_match.group(1))
        pressure_data["excess"] = float(pressure_match.group(2))
        pressure_data["total"] = float(pressure_match.group(3))

    return pressure_data


def parse_enthalpy_data(content: str) -> dict[str, dict[str, float]]:
    """
    Parse enthalpy of adsorption data from RASPA output content.

    Args:
        content: Text content of the RASPA output file.

    Returns:
        Dictionary with enthalpy data for each component.
    """
    enthalpy_data = {}

    # Extract component enthalpy sections
    enthalpy_pattern = r"Component\s+\d+\s+\[(.*?)\].*?Enthalpy of adsorption:\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\+/-\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\[K\]"
    enthalpy_matches = re.findall(enthalpy_pattern, content, re.DOTALL)

    for comp_name, value, error in enthalpy_matches:
        comp_name = comp_name.strip()
        enthalpy_data[comp_name] = {"value": float(value), "error": float(error)}

    return enthalpy_data


def parse_raspa_txt(file_path: str | Path) -> dict[str, Any]:
    """
    Parse a RASPA3 text output file.

    Args:
        file_path: Path to the text output file.

    Returns:
        Dictionary containing structured parsed data.
    """
    file_path = Path(file_path)

    with open(file_path) as f:
        content = f.read()

    # Extract basic information
    result = {
        "general": {},
        "adsorption": {},
        "energy": {},
        "pressure": {},
        "enthalpy": {},
    }

    # Extract version
    version_match = re.search(r"RASPA\s+(\d+\.\d+\.\d+)", content)
    if version_match:
        result["general"]["version"] = version_match.group(1)

    # Extract temperature and pressure
    temp_match = re.search(r"Temperature:\s+(\d+(?:\.\d+)?)\s+\[K\]", content)
    pressure_match = re.search(
        r"Pressure:\s+(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\[Pa\]", content
    )
    if temp_match:
        result["general"]["temperature"] = float(temp_match.group(1))
    if pressure_match:
        result["general"]["pressure"] = float(pressure_match.group(1))

    # Parse specific data sections
    result["adsorption"] = parse_adsorption_data(content)
    result["energy"] = parse_energy_data(content)
    result["pressure"] = parse_pressure_data(content)
    result["enthalpy"] = parse_enthalpy_data(content)

    return result


def extract_adsorption_data(parsed_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract adsorption data from parsed RASPA output.

    Args:
        parsed_data: Dictionary containing parsed RASPA data.

    Returns:
        Dictionary with adsorption data for each component.
    """
    # For JSON data
    if "properties" in parsed_data and "loadings" in parsed_data["properties"]:
        return parsed_data["properties"]["loadings"]

    # For text output
    if "adsorption" in parsed_data:
        return parsed_data["adsorption"]

    return {}


def extract_energy_data(parsed_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract energy data from parsed RASPA output.

    Args:
        parsed_data: Dictionary containing parsed RASPA data.

    Returns:
        Dictionary with energy components.
    """
    # For JSON data
    if "output" in parsed_data and "runningEnergies" in parsed_data["output"]:
        return parsed_data["output"]["runningEnergies"]

    # For text output
    if "energy" in parsed_data:
        return parsed_data["energy"]

    return {}


def extract_enthalpy_data(parsed_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract enthalpy of adsorption data from parsed RASPA output.

    Args:
        parsed_data: Dictionary containing parsed RASPA data.

    Returns:
        Dictionary with enthalpy data for each component.
    """
    # For JSON data
    if "properties" in parsed_data and "averageEnthalpy" in parsed_data["properties"]:
        enthalpy_data = {}
        for comp_name, data in parsed_data["properties"]["averageEnthalpy"].items():
            if "mean" in data:
                enthalpy_data[comp_name] = data["mean"]
        return enthalpy_data

    # For text output
    if "enthalpy" in parsed_data:
        return parsed_data["enthalpy"]

    return {}


def parse_output_directory(directory_path: str | Path) -> dict[str, Any]:
    """
    Parse all RASPA output files in a directory.

    Args:
        directory_path: Path to the directory containing RASPA output files.

    Returns:
        Dictionary with parsed data from all files.
    """
    directory_path = Path(directory_path)

    result = {"text_outputs": {}, "json_outputs": {}}

    # Look for .txt files
    txt_files = list(directory_path.glob("*.txt"))
    for txt_file in txt_files:
        try:
            result["text_outputs"][txt_file.name] = parse_raspa_txt(txt_file)
        except Exception as e:
            print(f"Error parsing {txt_file}: {e}")

    # Look for .json files
    json_files = list(directory_path.glob("*.json"))
    for json_file in json_files:
        try:
            result["json_outputs"][json_file.name] = parse_raspa_json(json_file)
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")

    return result
