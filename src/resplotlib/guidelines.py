import html
import json
from typing import Dict

from IPython.display import HTML, display

from . import utils


class Guidelines(Dict):
    """Guidelines

    This class is a dictionary that contains the plotting guidelines. It can be initialised with a dictionary or with a json file containing the guidelines.

    Args:
        *args: Arguments to initialise the dictionary.
        file_path (str, optional): Path to a json file containing the guidelines. Defaults to None.
    """

    def __init__(self, *args, file_path: str | None = None) -> None:
        """Initialise the Guidelines.

        Args:
            *args: Arguments to initialise the dictionary.
            file_path (str, optional): Path to a json file containing the guidelines. Defaults to None.
        """
        # If file_path is provided, read guidelines. Otherwise, initialise the dictionary with the provided arguments.
        if file_path is not None:
            guidelines = self._read_guidelines(file_path)
            super().__init__(**guidelines)
        else:
            super().__init__(*args)

    def _substitute_properties(self) -> None:
        """Substitute inherited and project properties in the guidelines."""

        # Substitute inherited properties
        self = utils._substitute_inherit_str_in_dicts(self)

        # Get project properties
        project_properties = self.get("project", {})

        # Substitute project properties
        for key, value in project_properties.items():
            self = utils._substitute_str_in_dict(self, f"@{key}", str(value))

    def _read_guidelines(self, file_path: str) -> Dict:
        """Read guidelines from a  file containing the guidelines.

        Args:
            file_path (str): Path to the json file containing the guidelines.

        Returns:
            dict: Dictionary with guidelines.
        """
        # Read guidelines from a json file containing the guidelines.
        with open(file_path, "r") as f:
            guidelines = json.load(f)

        return guidelines

    def _show_guideline_levels(self, _dict: Dict, level: int, indent: int) -> str:
        """Recursively show guideline levels.

        Args:
            _dict (dict): Guideline dictionary at the current level.
            level (int): Current indentation level.
            indent (int): Number of pixels to indent per level.

        Returns:
            str: HTML string representing the guidelines at this level.
        """
        # Initialize HTML string
        html_str = ""

        # Loop through dictionary items
        for key, value in _dict.items():
            safe_key = html.escape(str(key))
            if isinstance(value, dict):
                html_str += f"""
                <details>
                    <summary><strong>{safe_key}</strong></summary>
                    <div style="margin-left: {indent}px">
                        {self._show_guideline_levels(value, level + 1, indent)}
                    </div>
                </details>
                """
            else:
                safe_value = html.escape(str(value))
                html_str += f"""
                <div>
                    <strong>{safe_key}:</strong> {safe_value}
                </div>
                """

        return html_str

    def show(self, open: bool = True) -> None:
        """Show guidelines.

        Args:
            open (bool): Show guidelines as open or closed. Defaults to True.
        """
        # Set open/closed state
        open = "open" if open else "closed"

        # Set indent per level
        indent = 20

        # Show guidelines
        html_str = f"""
        <details {open}>
            <summary><strong>Guidelines</strong></summary>
            <div style="margin-left: {indent}px">
                {self._show_guideline_levels(self, 0, indent)}
            </div>
        </details>
        """
        display(HTML(html_str))
