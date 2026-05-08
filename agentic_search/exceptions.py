class AgenticSearchError(Exception):
    """Base exception for the framework."""


class ModelLoadError(AgenticSearchError):
    pass


class ModelResponseError(AgenticSearchError):
    pass


class SkillExecutionError(AgenticSearchError):
    pass


class ActionParseError(AgenticSearchError):
    pass


class ApiError(AgenticSearchError):
    pass


class ImagePreparationError(AgenticSearchError):
    pass
