import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketSink:
    def __init__(self, web_socket: WebSocket):
        self.ws = web_socket

    async def close(self):
        await self.ws.close(code=4000, reason="opened in another tab")

    async def send_frame(self, message):
        logger.debug("Sending frame")
        await self.ws.send_json(message)

    # def handle_connection(self, snapshot: dict):
    #     ws = self.ws
    #     if ws is not None:
    #         try:
    #             ws.send_json(snapshot)
    #         except Exception:
    #             self.game_session.pause()
