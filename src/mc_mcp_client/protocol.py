"""Protocol message types (CLIENT-03)."""


class ToolCallMessage:
    """Client → Server: invoke a tool."""
    # TODO: implement in CLIENT-03


class SynthesisMessage:
    """Client → Server: submit a synthesis."""
    # TODO: implement in CLIENT-03


class EpisodeEndMessage:
    """Client → Server: signal end of episode."""
    # TODO: implement in CLIENT-03


class SessionReadyMessage:
    """Server → Client: session established."""
    # TODO: implement in CLIENT-03


class ToolResultMessage:
    """Server → Client: result of a tool call."""
    # TODO: implement in CLIENT-03


class SynthesisRequiredMessage:
    """Server → Client: synthesis checkpoint triggered."""
    # TODO: implement in CLIENT-03


class SynthesisScoredMessage:
    """Server → Client: synthesis score returned."""
    # TODO: implement in CLIENT-03


class EpisodeCompleteMessage:
    """Server → Client: episode finished with final reward."""
    # TODO: implement in CLIENT-03


class ErrorMessage:
    """Server → Client: protocol or service error."""
    # TODO: implement in CLIENT-03
