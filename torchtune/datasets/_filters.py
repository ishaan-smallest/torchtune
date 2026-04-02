"""Dataset filter functions for chat datasets."""


def valid_chat_messages(sample: dict, messages_key: str = "messages") -> bool:
    """Filter out samples with invalid message ordering (e.g. consecutive
    assistant messages without an intervening user/tool message)."""
    msgs = sample.get(messages_key, [])
    if len(msgs) < 2:
        return len(msgs) > 0
    for i in range(1, len(msgs)):
        if msgs[i]["role"] == msgs[i - 1]["role"] == "assistant":
            return False
    return True
