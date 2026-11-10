"""Slack-based approval backend using Block Kit buttons over Socket Mode.

Socket Mode (an app-level WebSocket), not the HTTP Events API -- this
backend never needs a public webhook URL, matching the Discord backend's
own persistent-connection shape rather than requiring schwab-mcp's host
to be internet-reachable.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

from schwab_mcp.approvals.base import (
    ApprovalDecision,
    ApprovalManager,
    ApprovalRequest,
    format_arguments,
    parse_account_names,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SlackApprovalSettings:
    """Configuration values required for Slack approvals."""

    bot_token: str
    app_token: str
    channel_id: str
    approver_ids: frozenset[str] = frozenset()
    timeout_seconds: float = 600.0
    account_names: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    """Map the last four characters of an account hash to a friendly name."""


@dataclass(slots=True)
class _PendingApproval:
    request: ApprovalRequest
    future: asyncio.Future[ApprovalDecision]
    channel: str
    message_ts: str


class SlackApprovalManager(ApprovalManager):
    """Approval manager that routes decisions through a Slack message's
    Approve/Deny buttons, received over Socket Mode."""

    def __init__(self, settings: SlackApprovalSettings) -> None:
        if not settings.approver_ids:
            raise ValueError("SlackApprovalManager requires at least one approver ID.")

        self._settings = settings
        self._web_client = AsyncWebClient(token=settings.bot_token)
        self._socket_client: SocketModeClient | None = None
        self._pending: dict[str, _PendingApproval] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Connect the Socket Mode client. ``SocketModeClient.connect()``
        establishes the websocket and spawns its own background
        receive/monitor tasks, then returns -- it does not block, unlike
        the Discord backend's gateway client, so there's no separate
        runner task to manage here."""
        if self._socket_client is not None:
            return

        client = SocketModeClient(app_token=self._settings.app_token, web_client=self._web_client)
        client.socket_mode_request_listeners.append(self._handle_request)
        await client.connect()
        self._socket_client = client

    async def stop(self) -> None:
        if self._socket_client is None:
            return
        await self._socket_client.close()
        self._socket_client = None

    async def require(self, request: ApprovalRequest) -> ApprovalDecision:
        """Post an approval request to Slack and wait for a button click."""
        await self.start()

        blocks = self._build_pending_blocks(request)
        try:
            response = await self._web_client.chat_postMessage(
                channel=self._settings.channel_id,
                text=f"Write operation requires approval: {request.tool_name}",
                blocks=blocks,
            )
        except SlackApiError:
            logger.exception("Failed to post Slack approval request for request %s", request.id)
            return ApprovalDecision.DENIED

        message_ts = response["ts"]
        future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        pending = _PendingApproval(
            request=request, future=future, channel=self._settings.channel_id, message_ts=message_ts
        )

        async with self._lock:
            self._pending[message_ts] = pending

        try:
            decision = await asyncio.wait_for(future, timeout=self._settings.timeout_seconds)
        except asyncio.TimeoutError:
            decision = ApprovalDecision.EXPIRED
            await self._finalize_message(
                pending,
                decision,
                actor_id=None,
                reason=f"No decision within {int(self._settings.timeout_seconds)}s timeout.",
            )
        finally:
            async with self._lock:
                self._pending.pop(message_ts, None)

        return decision

    async def _handle_request(self, client: SocketModeClient, req: SocketModeRequest) -> None:
        # Every socket-mode request must be acked, regardless of type,
        # or Slack will retry delivery.
        await client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

        if req.type != "interactive":
            return
        payload = req.payload
        if payload.get("type") != "block_actions":
            return
        actions = payload.get("actions") or []
        if not actions:
            return
        action_id = actions[0].get("action_id")
        if action_id not in ("approve", "deny"):
            return

        message_ts = (payload.get("message") or {}).get("ts")
        user_id = (payload.get("user") or {}).get("id")

        async with self._lock:
            pending = self._pending.get(message_ts)
        if pending is None:
            return

        if user_id not in self._settings.approver_ids:
            logger.debug(
                "Ignoring Slack action %s from unauthorized user %s for request %s",
                action_id,
                user_id,
                pending.request.id,
            )
            return

        if pending.future.done():
            return

        decision = ApprovalDecision.APPROVED if action_id == "approve" else ApprovalDecision.DENIED
        pending.future.set_result(decision)
        await self._finalize_message(pending, decision, actor_id=user_id, reason=f"Decision recorded via {action_id}")

    def _build_pending_blocks(self, request: ApprovalRequest) -> list[dict]:
        lines = ["*Write operation requires approval*", f"Tool: `{request.tool_name}`"]
        account_label = self._account_label(request.arguments)
        if account_label:
            lines.append(f"Account: {account_label}")
        lines.append(f"Request ID: {request.request_id}")
        lines.append(f"Approval ID: {request.id}")
        if request.client_id:
            lines.append(f"Client ID: {request.client_id}")

        blocks: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}}]
        if request.arguments:
            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": format_arguments(request.arguments)}}
            )
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": "approve",
                        "value": request.id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Deny"},
                        "style": "danger",
                        "action_id": "deny",
                        "value": request.id,
                    },
                ],
            }
        )
        return blocks

    async def _finalize_message(
        self,
        pending: _PendingApproval,
        decision: ApprovalDecision,
        *,
        actor_id: str | None,
        reason: str | None,
    ) -> None:
        lines = [f"*Write operation {decision.value}*", f"Tool: `{pending.request.tool_name}`"]
        account_label = self._account_label(pending.request.arguments)
        if account_label:
            lines.append(f"Account: {account_label}")
        lines.append(f"Request ID: {pending.request.request_id}")
        lines.append(f"Approval ID: {pending.request.id}")
        if pending.request.client_id:
            lines.append(f"Client ID: {pending.request.client_id}")
        if pending.request.arguments:
            lines.append(format_arguments(pending.request.arguments))
        if actor_id is not None:
            lines.append(f"Actor: <@{actor_id}>")
        if reason:
            lines.append(f"Notes: {reason}")

        try:
            await self._web_client.chat_update(
                channel=pending.channel,
                ts=pending.message_ts,
                text=f"Write operation {decision.value}: {pending.request.tool_name}",
                blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}}],
            )
        except SlackApiError:
            logger.warning("Failed to update Slack approval message for request %s", pending.request.id, exc_info=True)

    def _account_label(self, arguments: Mapping[str, str]) -> str | None:
        if not self._settings.account_names:
            return None

        account_hash = arguments.get("account_hash")
        if not account_hash:
            return None

        # Cancellation requests use repr(); previewed orders use plain strings.
        if account_hash.startswith(("'", '"')):
            try:
                account_hash = ast.literal_eval(account_hash)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                return None
        if not isinstance(account_hash, str) or not account_hash.isascii() or not account_hash.isalnum():
            return None

        suffix = account_hash[-4:]
        return self._settings.account_names.get(suffix) or f"…{suffix} (no configured name)"

    @staticmethod
    def parse_account_names(values: Sequence[str] | None) -> Mapping[str, str]:
        """Parse configured account names."""
        return parse_account_names(values)

    @staticmethod
    def authorized_user_ids(users: Sequence[str] | None) -> frozenset[str]:
        """Normalize a sequence of authorized Slack user IDs. Unlike
        Discord's numeric snowflake IDs, Slack user IDs (e.g.
        ``U0BFQBQ57A6``) are always strings -- never coerced to int."""
        if not users:
            return frozenset()
        return frozenset(str(user) for user in users)


__all__ = ["SlackApprovalManager", "SlackApprovalSettings"]
