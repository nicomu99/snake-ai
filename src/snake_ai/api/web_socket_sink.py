"""Sends game frames to a client over a WebSocket."""

import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketSink:
    """A thin wrapper around a WebSocket that a ``GameRunner`` sends frames to.

    Args:
        web_socket: An accepted WebSocket connection.
    """

    def __init__(self, web_socket: WebSocket):
        self.ws = web_socket

    async def close(self):
        """Closes the socket because the game was opened somewhere else.

        Uses close code 4000 with the reason "opened in another tab".
        """
        await self.ws.close(code=4000, reason="opened in another tab")

    async def send_frame(self, message):
        """Sends a message to the client as JSON.

        Args:
            message: A JSON-serializable dict.
        """
        logger.debug("Sending frame")
        await self.ws.send_json(message)

    # def handle_connection(self, snapshot: dict):
    #     ws = self.ws
    #     if ws is not None:
    #         try:
    #             ws.send_json(snapshot)
    #         except Exception:
    #             self.game_session.pause()
