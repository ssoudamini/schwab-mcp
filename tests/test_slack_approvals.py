"""Unit tests for schwab_mcp/approvals/slack.py."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from slack_sdk.errors import SlackApiError

from schwab_mcp.approvals.base import ApprovalDecision, ApprovalRequest
from schwab_mcp.approvals.slack import SlackApprovalManager, SlackApprovalSettings

APPROVER_ID = "U_APPROVER"
OTHER_USER_ID = "U_OTHER"
CHANNEL_ID = "C_TARGET"
MESSAGE_TS = "1234.5678"


def make_settings(
    *,
    approver_ids: frozenset[str] = frozenset({APPROVER_ID}),
    timeout_seconds: float = 5.0,
) -> SlackApprovalSettings:
    return SlackApprovalSettings(
        bot_token="xoxb-fake",
        app_token="xapp-fake",
        channel_id=CHANNEL_ID,
        approver_ids=approver_ids,
        timeout_seconds=timeout_seconds,
    )


def make_request(
    *,
    id: str = "approval-1",
    tool_name: str = "place_previewed_order",
    request_id: str = "req-1",
    client_id: str | None = "client-1",
    arguments: dict[str, str] | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        id=id,
        tool_name=tool_name,
        request_id=request_id,
        client_id=client_id,
        arguments=arguments if arguments is not None else {"symbol": "AAPL", "qty": "10"},
    )


def make_manager(settings: SlackApprovalSettings | None = None) -> SlackApprovalManager:
    """Return a manager with a mocked AsyncWebClient and a pre-connected
    fake socket client, so tests never touch the real network."""
    mgr = SlackApprovalManager(settings or make_settings())
    mgr._web_client = AsyncMock()
    mgr._web_client.chat_postMessage = AsyncMock(return_value={"ts": MESSAGE_TS})
    mgr._web_client.chat_update = AsyncMock()
    mgr._socket_client = MagicMock()  # start() already "connected"
    mgr._socket_client.send_socket_mode_response = AsyncMock()
    mgr._socket_client.close = AsyncMock()
    return mgr


def block_action_request(*, action_id: str, user_id: str, message_ts: str = MESSAGE_TS) -> Any:
    req = MagicMock()
    req.type = "interactive"
    req.envelope_id = "env-1"
    req.payload = {
        "type": "block_actions",
        "actions": [{"action_id": action_id}],
        "message": {"ts": message_ts},
        "user": {"id": user_id},
    }
    return req


async def wait_until_pending(mgr: SlackApprovalManager, message_ts: str = MESSAGE_TS) -> None:
    async def poll() -> None:
        while message_ts not in mgr._pending:
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1.0)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_requires_at_least_one_approver_id() -> None:
    with pytest.raises(ValueError, match="at least one approver ID"):
        SlackApprovalManager(make_settings(approver_ids=frozenset()))


# ---------------------------------------------------------------------------
# require(): success path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action_id", "expected"),
    [("approve", ApprovalDecision.APPROVED), ("deny", ApprovalDecision.DENIED)],
)
@pytest.mark.anyio
async def test_require_records_authorized_click(action_id: str, expected: ApprovalDecision) -> None:
    mgr = make_manager()

    async def submit_click() -> None:
        await wait_until_pending(mgr)
        await mgr._handle_request(mgr._socket_client, block_action_request(action_id=action_id, user_id=APPROVER_ID))

    click_task = asyncio.create_task(submit_click())
    decision = await mgr.require(make_request())
    await click_task

    assert decision == expected
    assert MESSAGE_TS not in mgr._pending
    mgr._web_client.chat_update.assert_awaited_once()
    update_kwargs = mgr._web_client.chat_update.await_args.kwargs
    assert expected.value in update_kwargs["text"]
    rendered = update_kwargs["blocks"][0]["text"]["text"]
    assert APPROVER_ID in rendered
    assert action_id in rendered


@pytest.mark.anyio
async def test_require_posts_message_with_tool_name() -> None:
    mgr = make_manager()

    async def resolve() -> None:
        await wait_until_pending(mgr)
        mgr._pending[MESSAGE_TS].future.set_result(ApprovalDecision.APPROVED)

    asyncio.get_running_loop().create_task(resolve())
    await mgr.require(make_request(tool_name="my_special_tool"))

    call_kwargs = mgr._web_client.chat_postMessage.await_args.kwargs
    assert call_kwargs["channel"] == CHANNEL_ID
    rendered = call_kwargs["blocks"][0]["text"]["text"]
    assert "my_special_tool" in rendered


# ---------------------------------------------------------------------------
# require(): timeout / unauthorized path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_id", "action_id"),
    [
        pytest.param(OTHER_USER_ID, "approve", id="unauthorized-user"),
        pytest.param(APPROVER_ID, "snooze", id="unsupported-action"),
    ],
)
@pytest.mark.anyio
async def test_invalid_click_cannot_approve_live_request(user_id: str, action_id: str) -> None:
    mgr = make_manager(make_settings(timeout_seconds=0.05))

    async def submit_invalid_click() -> None:
        await wait_until_pending(mgr)
        await mgr._handle_request(mgr._socket_client, block_action_request(action_id=action_id, user_id=user_id))

    click_task = asyncio.create_task(submit_invalid_click())
    decision = await asyncio.wait_for(mgr.require(make_request()), timeout=2.0)
    await click_task

    assert decision == ApprovalDecision.EXPIRED
    assert MESSAGE_TS not in mgr._pending
    mgr._web_client.chat_update.assert_awaited_once()
    update_kwargs = mgr._web_client.chat_update.await_args.kwargs
    assert "expired" in update_kwargs["text"]
    assert "timeout" in update_kwargs["blocks"][0]["text"]["text"]


@pytest.mark.anyio
async def test_require_denied_when_post_message_raises() -> None:
    mgr = make_manager()
    mgr._web_client.chat_postMessage = AsyncMock(
        side_effect=SlackApiError("boom", MagicMock(status_code=500, data={"error": "boom"}))
    )

    decision = await mgr.require(make_request())

    assert decision == ApprovalDecision.DENIED
    assert mgr._pending == {}


@pytest.mark.anyio
async def test_require_keeps_approval_when_message_update_fails() -> None:
    mgr = make_manager()
    mgr._web_client.chat_update = AsyncMock(
        side_effect=SlackApiError("boom", MagicMock(status_code=500, data={"error": "boom"}))
    )

    async def submit_click() -> None:
        await wait_until_pending(mgr)
        await mgr._handle_request(mgr._socket_client, block_action_request(action_id="approve", user_id=APPROVER_ID))

    click_task = asyncio.create_task(submit_click())
    decision = await asyncio.wait_for(mgr.require(make_request()), timeout=2.0)
    await click_task

    assert decision == ApprovalDecision.APPROVED


# ---------------------------------------------------------------------------
# _handle_request(): routing/ack logic
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_every_request_is_acked_regardless_of_type() -> None:
    mgr = make_manager()
    req = MagicMock()
    req.type = "events_api"
    req.envelope_id = "env-2"
    req.payload = {}

    await mgr._handle_request(mgr._socket_client, req)

    mgr._socket_client.send_socket_mode_response.assert_awaited_once()


@pytest.mark.anyio
async def test_handle_request_ignores_unknown_message_ts() -> None:
    mgr = make_manager()
    # Nothing registered in _pending -- must complete silently, not raise.
    await mgr._handle_request(mgr._socket_client, block_action_request(action_id="approve", user_id=APPROVER_ID))


@pytest.mark.anyio
async def test_handle_request_skips_already_resolved_future() -> None:
    mgr = make_manager()
    future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
    future.set_result(ApprovalDecision.APPROVED)
    from schwab_mcp.approvals.slack import _PendingApproval

    mgr._pending[MESSAGE_TS] = _PendingApproval(
        request=make_request(), future=future, channel=CHANNEL_ID, message_ts=MESSAGE_TS
    )

    await mgr._handle_request(mgr._socket_client, block_action_request(action_id="approve", user_id=APPROVER_ID))

    mgr._web_client.chat_update.assert_not_awaited()


# ---------------------------------------------------------------------------
# start() / stop() lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_start_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = SlackApprovalManager(make_settings())
    fake_client = MagicMock()
    fake_client.connect = AsyncMock()
    fake_client.socket_mode_request_listeners = []

    monkeypatch.setattr("schwab_mcp.approvals.slack.SocketModeClient", lambda **kwargs: fake_client)

    await mgr.start()
    first_client = mgr._socket_client
    await mgr.start()  # second call must be a no-op

    assert mgr._socket_client is first_client
    fake_client.connect.assert_awaited_once()


@pytest.mark.anyio
async def test_stop_clears_state() -> None:
    mgr = make_manager()
    mgr._socket_client.close = AsyncMock()

    await mgr.stop()

    assert mgr._socket_client is None


@pytest.mark.anyio
async def test_stop_when_not_started_is_safe() -> None:
    mgr = SlackApprovalManager(make_settings())
    await mgr.stop()  # must not raise


# ---------------------------------------------------------------------------
# authorized_user_ids() static helper
# ---------------------------------------------------------------------------


def test_authorized_user_ids_normalizes_sequence() -> None:
    assert SlackApprovalManager.authorized_user_ids(["U1", "U2"]) == frozenset({"U1", "U2"})


def test_authorized_user_ids_returns_empty_frozenset_for_none() -> None:
    assert SlackApprovalManager.authorized_user_ids(None) == frozenset()


def test_authorized_user_ids_keeps_ids_as_strings_not_ints() -> None:
    """Unlike Discord's numeric snowflake IDs, Slack user IDs
    (e.g. 'U0BFQBQ57A6') are never valid integers -- confirms this
    helper never attempts an int() coercion the way Discord's does."""
    result = SlackApprovalManager.authorized_user_ids(["U0BFQBQ57A6"])
    assert result == frozenset({"U0BFQBQ57A6"})


# ---------------------------------------------------------------------------
# Account name labeling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "expected"), [("A1B2", "Example Brokerage"), ("C3D4", "…A1B2 (no configured name)")]
)
def test_account_label_present_or_falls_back(suffix: str, expected: str) -> None:
    mgr = SlackApprovalManager(
        SlackApprovalSettings(
            bot_token="xoxb-fake",
            app_token="xapp-fake",
            channel_id=CHANNEL_ID,
            approver_ids=frozenset({APPROVER_ID}),
            account_names=SlackApprovalManager.parse_account_names([f"{suffix}=Example Brokerage"]),
        )
    )
    label = mgr._account_label({"account_hash": "0123456789ABCDEFA1B2"})
    assert label == expected


def test_account_label_none_without_configured_names() -> None:
    mgr = SlackApprovalManager(make_settings())
    assert mgr._account_label({"account_hash": "0123456789ABCDEFA1B2"}) is None


def test_account_label_none_without_account_hash_argument() -> None:
    mgr = SlackApprovalManager(
        SlackApprovalSettings(
            bot_token="xoxb-fake",
            app_token="xapp-fake",
            channel_id=CHANNEL_ID,
            approver_ids=frozenset({APPROVER_ID}),
            account_names=SlackApprovalManager.parse_account_names(["A1B2=Example Brokerage"]),
        )
    )
    assert mgr._account_label({}) is None


# ---------------------------------------------------------------------------
# Structured argument rendering
# ---------------------------------------------------------------------------


def test_render_argument_blocks_renders_known_fields_as_a_grid() -> None:
    mgr = make_manager()
    blocks = mgr._render_argument_blocks(
        {
            "ticker": "AAPL",
            "action": "BUY",
            "quantity": "10",
            "order_type": "LIMIT",
            "price": "150.00",
            "session": "NORMAL",
            "duration": "DAY",
            "bid": "149.95",
            "ask": "150.05",
            "mid": "150.00",
        }
    )
    field_block = blocks[0]
    assert field_block["type"] == "section"
    texts = [f["text"] for f in field_block["fields"]]
    assert "*Ticker:*\nAAPL" in texts
    assert "*Action:*\nBUY" in texts
    assert "*Quantity:*\n10" in texts
    assert "*Order Type:*\nLIMIT" in texts
    assert "*Limit Price:*\n150.00" in texts
    assert "*Session:*\nNORMAL" in texts
    assert "*Duration:*\nDAY" in texts
    assert "*Bid:*\n149.95" in texts
    assert "*Ask:*\n150.05" in texts
    assert "*Mid:*\n150.00" in texts


def test_render_argument_blocks_includes_rationale_and_summary() -> None:
    mgr = make_manager()
    blocks = mgr._render_argument_blocks(
        {"order_summary": "BUY 10 AAPL LIMIT @ 150.00", "rationale": "Hedging earnings risk."}
    )
    texts = [b["text"]["text"] for b in blocks if "text" in b]
    assert any("Hedging earnings risk." in t for t in texts)
    assert any("BUY 10 AAPL LIMIT @ 150.00" in t for t in texts)


def test_render_argument_blocks_omits_rationale_when_absent() -> None:
    mgr = make_manager()
    blocks = mgr._render_argument_blocks({"order_summary": "BUY 10 AAPL LIMIT @ 150.00"})
    texts = [b["text"]["text"] for b in blocks if "text" in b]
    assert not any("Rationale" in t for t in texts)


def test_render_argument_blocks_falls_back_for_multi_leg_orders() -> None:
    """A multi-leg order (OCO/trigger/bracket/combo) has no single ticker
    to break out -- it should still render via the generic fallback dump,
    not silently drop its summary/bookkeeping fields."""
    mgr = make_manager()
    blocks = mgr._render_argument_blocks(
        {"original_tool": "preview_oco_order", "order_summary": "OCO: LIMIT 150 / STOP 140", "preview_id": "abc123"}
    )
    assert not any(b.get("fields") for b in blocks)
    texts = [b["text"]["text"] for b in blocks if "text" in b]
    assert any("OCO: LIMIT 150 / STOP 140" in t for t in texts)
    assert any("preview_id" in t for t in texts)


def test_render_argument_blocks_excludes_account_hash_from_fallback_dump() -> None:
    """account_hash is rendered separately via _account_label, not dumped
    again in the generic fallback section."""
    mgr = make_manager()
    blocks = mgr._render_argument_blocks({"order_summary": "x", "account_hash": "0123456789ABCDEF5805"})
    texts = [b["text"]["text"] for b in blocks if "text" in b]
    assert not any("account_hash" in t for t in texts)


def test_render_argument_blocks_empty_arguments_yields_no_blocks() -> None:
    mgr = make_manager()
    assert mgr._render_argument_blocks({}) == []
