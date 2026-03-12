"""This file contains the graph utilities for the application."""
from pathlib import Path
from app.core.config import settings


def get_vector_path(user_id: str, session_id: str) -> Path:
        """Sanitize the file path
        raises:
            - ValueError: If any other paths are given
        """
        vector_dir_path = (settings.VECTOR_FOLDER / user_id / session_id).resolve()
        if not vector_dir_path.is_relative_to(settings.VECTOR_FOLDER):
            raise ValueError(
                f"Vector file address must be within the limit.Path=> {vector_dir_path}"
            )
        return vector_dir_path
