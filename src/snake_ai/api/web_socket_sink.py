from fastapi import WebSocket

from snake_ai.snake import GameSession, Model
from snake_ai.ai import Agent


class WebSocketSink:
    def __init__(self, web_socket: WebSocket):
        self.ws = web_socket

    def close(self):
        self.ws.close(code=4000, reason="opened in another tab")

    async def send_frame(self, message):
        await self.ws.send_json(message)

    def handle_connection(self, snapshot: dict):
        ws = self.ws
        if ws is not None:
            try:
                ws.send_json(snapshot)
            except Exception:
                self.game_session.pause()

