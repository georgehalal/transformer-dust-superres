import yaml


def read_yaml_file(file_path: str) -> dict:
    """Read a YAML file and return its contents as a dictionary.

    Parameters:
        file_path: Path to the YAML file.

    Returns:
        dict: Contents of the YAML file.
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
