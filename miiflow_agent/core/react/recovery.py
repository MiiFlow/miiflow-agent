"""Graduated recovery strategies for tool and LLM errors.

Instead of simply counting consecutive errors and stopping, the RecoveryManager
tries progressively more aggressive recovery strategies before giving up.
Inspired by Claude Code's multi-level retry with different recovery approaches.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from ..message import Message, MessageRole

logger = logging.getLogger(__name__)


# Substrings used to detect context-overflow / max-tokens errors across
# providers. Each provider raises its own exception class with its own message,
# so we sniff strings as a portable lowest-common-denominator. The check is
# best-effort: a false positive only means we try compaction once before
# falling through to the normal recovery ladder.
_CONTEXT_OVERFLOW_HINTS = (
    "context length",
    "context_length",
    "context window",
    "maximum context",
    "max_tokens",
    "max output tokens",
    "max_output_tokens",
    "too many tokens",
    "prompt is too long",
    "input is too long",
    "exceeded the context",
    "string too long",
    "request too large",
)


# Provider errors no ladder step can change: the request will fail the same
# way on retry-with-guidance, after compaction and with fewer tools, because
# the ACCOUNT — not the request — is what is wrong. Matched on the provider's
# own wording (the ladder receives ``Exception(step.error)``, a string wrapper,
# so the SDK exception class is gone by the time it gets here).
_FATAL_PROVIDER_HINTS = (
    # Anthropic billing
    "credit balance is too low",
    "plans & billing",
    # OpenAI billing / quota
    "exceeded your current quota",
    "insufficient_quota",
    "billing_hard_limit_reached",
    "billing hard limit",
    # Auth / permission (any provider)
    "authentication_error",
    "invalid x-api-key",
    "invalid api key",
    "incorrect api key",
    "api key not valid",
    "permission_error",
    "permission denied",
    "does not have access to model",
    "unauthorized",
    # Deployment / model configuration
    "model_not_found",
    "not_found_error",
    "does not exist or you do not have access",
)


def is_fatal_provider_error(error: BaseException) -> bool:
    """True when the provider rejected the request for a reason no retry fixes.

    2026-08-11 07:21 UTC: the Anthropic account ran out of credit and every
    agent in the suggestion run answered the 400 by walking the full recovery
    ladder — five identical requests per agent inside one second, forty
    failing calls across six orgs, then the canned "repeated issues" answer.
    Guidance, compaction and tool simplification all address the REQUEST;
    a billing or auth failure is about the ACCOUNT, so the only correct move
    is to stop at once (and let the halt reason name the cause).
    """
    if error is None:
        return False
    text = (str(error) or "").lower()
    return any(hint in text for hint in _FATAL_PROVIDER_HINTS)


def is_tool_approval_error(error: BaseException) -> bool:
    """True if ``error`` is (or stringifies to) a tool-approval pause.

    Approval pauses are control flow, not failures: the only resolution is the
    user approving the tool, so retrying with "try a different approach" or
    compacting the context just burns the recovery ladder and corrupts the
    transcript. The batch tool path now pauses correctly before recovery ever
    runs (see ``_handle_parallel_tool_batch``); this is a belt-and-suspenders
    guard for any path that surfaces the pause as a generic step error.
    """
    if error is None:
        return False
    if type(error).__name__ == "ToolApprovalRequired":
        return True
    return "requires user approval" in (str(error) or "").lower()


def is_context_overflow_error(error: BaseException) -> bool:
    """Heuristically classify ``error`` as a context-overflow / token-limit error.

    Works across providers (Anthropic, OpenAI, Gemini, Groq, ...) by matching
    common phrases in the exception message and class name. Returns False for
    None or unrelated errors.
    """
    if error is None:
        return False
    text = (str(error) or "").lower()
    cls = type(error).__name__.lower()
    if any(hint in text for hint in _CONTEXT_OVERFLOW_HINTS):
        return True
    if "contextlength" in cls or "tokenlimit" in cls:
        return True
    return False


class RecoveryStrategy(Enum):
    """Available recovery strategies, ordered by aggressiveness."""

    RETRY_WITH_GUIDANCE = "guidance"  # Add error context to help LLM adjust
    COMPRESS_AND_RETRY = "compress"  # Compress context, then retry
    SIMPLIFY_TOOLS = "simplify"  # Exclude the failing tool


class FailureKind(Enum):
    """Why a step failed. Different kinds gate different recovery behaviors."""

    # The tool itself raised, the API returned 5xx, etc. The tool may genuinely
    # be broken or unsuitable — eligible for the SIMPLIFY_TOOLS escalation.
    RUNTIME = "runtime"

    # The model called the tool with an arg shape that didn't match the schema
    # (e.g. missing required fields). The tool is fine; the LLM needs better
    # feedback. The orchestrator emits a structured tool_use_error itself, so
    # recovery should NOT add a duplicate guidance message and must NOT count
    # this against the per-tool exclusion threshold.
    SCHEMA = "schema"

    # The model's tool_use block was truncated mid-stream by max_tokens. The
    # tool is fine; the args are incomplete. Same handling as SCHEMA.
    TRUNCATION = "truncation"


# Default strategy sequence
DEFAULT_STRATEGIES = [
    RecoveryStrategy.RETRY_WITH_GUIDANCE,
    RecoveryStrategy.COMPRESS_AND_RETRY,
    RecoveryStrategy.SIMPLIFY_TOOLS,
]


@dataclass
class RecoveryAction:
    """Describes what to do after a recovery attempt."""

    strategy_used: RecoveryStrategy
    should_continue: bool  # True = retry the step, False = stop execution
    guidance_message: Optional[str] = None  # Added to context as user message
    excluded_tools: Optional[Set[str]] = None  # Tools to remove from pool
    attempt_number: int = 0


class RecoveryManager:
    """Graduated recovery strategies for agent errors.

    Instead of the simple "3 consecutive errors = stop" approach, this manager
    tries different recovery strategies before giving up:

    1. RETRY_WITH_GUIDANCE: Add a message explaining the error and asking
       the LLM to try a different approach.
    2. COMPRESS_AND_RETRY: If context might be causing confusion, compress it.
    3. SIMPLIFY_TOOLS: If the same tool keeps failing, exclude it and let
       the LLM work with remaining tools.
    """

    def __init__(
        self,
        max_recovery_attempts: int = 3,
        strategies: Optional[List[RecoveryStrategy]] = None,
        context_compressor=None,
        compress_fn=None,
    ):
        """Initialize recovery manager.

        Args:
            max_recovery_attempts: Maximum recovery attempts before stopping.
            strategies: Ordered list of strategies to try. Defaults to all three.
            context_compressor: Optional legacy ContextCompressor for
                COMPRESS_AND_RETRY. Must expose ``compress_if_needed``; a
                ``ContextEngine`` passed here is rejected loudly (the engine
                needs the full request shape, which only the orchestrator can
                build — wire ``compress_fn`` instead).
            compress_fn: Optional ``async (context) -> bool`` that performs the
                actual compaction. The orchestrator injects its own forced-
                compaction routine here so recovery decides *policy* while the
                component that knows the wire shape owns the *mechanism*.
                Takes precedence over ``context_compressor``.
        """
        self.max_recovery_attempts = max_recovery_attempts
        self.strategies = strategies or DEFAULT_STRATEGIES
        self.context_compressor = context_compressor
        self.compress_fn = compress_fn

        # The overflow fast path bypasses the normal attempt ladder (see
        # attempt_recovery), so it needs its own cap: when compaction cannot
        # bring the request under the wall, overflow → compact → overflow
        # would otherwise loop forever, burning a summarization call per turn.
        self.max_overflow_attempts = 3

        # Track state across attempts
        self._attempt_count: int = 0
        self._overflow_attempts: int = 0
        self._tool_error_counts: Dict[str, int] = {}
        self._excluded_tools: Set[str] = set()

    def reset(self):
        """Reset recovery state (call when a step succeeds)."""
        self._attempt_count = 0
        self._overflow_attempts = 0
        # Don't reset tool error counts - they accumulate across the session

    def record_success(self):
        """Record a successful step, resetting the attempt counter."""
        self._attempt_count = 0
        self._overflow_attempts = 0

    async def attempt_recovery(
        self,
        error: Exception,
        context,
        step=None,
        tool_name: Optional[str] = None,
        failure_kind: FailureKind = FailureKind.RUNTIME,
    ) -> RecoveryAction:
        """Determine recovery action based on error type and attempt number.

        Args:
            error: The exception that occurred.
            context: Current RunContext.
            step: The ReActStep that failed (optional).
            tool_name: Name of the tool that failed (optional).
            failure_kind: Whether this is a runtime, schema, or truncation
                failure. Schema/truncation failures don't count toward the
                per-tool exclusion threshold or the recovery ladder; the
                orchestrator already emitted a structured tool_use_error
                that the LLM will see on its next turn.

        Returns:
            RecoveryAction describing what to do next.
        """
        # SCHEMA / TRUNCATION: the orchestrator already pushed an actionable
        # tool_use_error into the message history. Don't double-prompt the
        # LLM and don't pollute the runtime failure counters that drive
        # tool exclusion. The safety_manager's max_iterations cap is what
        # protects us from infinite self-correction loops.
        if failure_kind in (FailureKind.SCHEMA, FailureKind.TRUNCATION):
            return RecoveryAction(
                strategy_used=RecoveryStrategy.RETRY_WITH_GUIDANCE,
                should_continue=True,
                guidance_message=None,
                attempt_number=self._attempt_count,
            )

        # Tool-approval pauses are not recoverable failures — stop cleanly
        # instead of retrying an action that can only proceed with user
        # approval. (The orchestrator's tool paths pause before recovery runs;
        # this catches any path that leaked the pause as a step error.)
        if is_tool_approval_error(error):
            logger.info(
                "Recovery received a tool-approval pause; stopping the loop "
                "(approval cannot be satisfied by a retry)."
            )
            return RecoveryAction(
                strategy_used=RecoveryStrategy.RETRY_WITH_GUIDANCE,
                should_continue=False,
                guidance_message=None,
                attempt_number=self._attempt_count,
            )

        # Account-level provider failures (billing, auth, missing model) are
        # not recoverable by anything below; retrying only multiplies the
        # failing calls. Stop now — the run's failure metadata carries the
        # cause. LLM-step failures only (`tool_name is None`): a TOOL that
        # answers "401 Unauthorized" is one integration's expired credential,
        # which the ladder handles by guidance and SIMPLIFY_TOOLS — it must
        # not halt the whole run.
        if tool_name is None and is_fatal_provider_error(error):
            logger.error(
                "Recovery received a fatal provider error; stopping the loop "
                "instead of retrying: %s",
                str(error)[:300],
            )
            return RecoveryAction(
                strategy_used=RecoveryStrategy.RETRY_WITH_GUIDANCE,
                should_continue=False,
                guidance_message=None,
                attempt_number=self._attempt_count,
            )

        self._attempt_count += 1

        # Track per-tool failures (runtime only)
        if tool_name:
            self._tool_error_counts[tool_name] = self._tool_error_counts.get(tool_name, 0) + 1

        # Context-overflow / token-limit errors are special: the only useful
        # response is to compact the conversation. Jump straight to the
        # compression strategy regardless of attempt index, so we don't waste
        # an attempt on RETRY_WITH_GUIDANCE that will hit the same wall.
        if is_context_overflow_error(error) and (
            self.compress_fn is not None or self.context_compressor is not None
        ):
            self._overflow_attempts += 1
            if self._overflow_attempts > self.max_overflow_attempts:
                logger.warning(
                    "Context-overflow recovery exhausted after %d compaction "
                    "attempts; compaction cannot bring the request under the "
                    "provider's limit. Stopping.",
                    self._overflow_attempts - 1,
                )
                return RecoveryAction(
                    strategy_used=RecoveryStrategy.COMPRESS_AND_RETRY,
                    should_continue=False,
                    attempt_number=self._attempt_count,
                )
            logger.info(
                "Detected context-overflow error; routing recovery to COMPRESS_AND_RETRY"
            )
            return await self._apply_compression(error, context, tool_name)

        # Exhausted all recovery attempts
        if self._attempt_count > self.max_recovery_attempts:
            logger.warning(
                f"Recovery exhausted after {self._attempt_count - 1} attempts. Stopping."
            )
            return RecoveryAction(
                strategy_used=self.strategies[-1] if self.strategies else RecoveryStrategy.RETRY_WITH_GUIDANCE,
                should_continue=False,
                attempt_number=self._attempt_count,
            )

        # Select strategy based on attempt number
        strategy_idx = min(self._attempt_count - 1, len(self.strategies) - 1)
        strategy = self.strategies[strategy_idx]

        logger.info(
            f"Recovery attempt {self._attempt_count}/{self.max_recovery_attempts}: "
            f"strategy={strategy.value}, error={str(error)[:200]}"
        )

        if strategy == RecoveryStrategy.RETRY_WITH_GUIDANCE:
            return self._apply_guidance(error, tool_name)

        elif strategy == RecoveryStrategy.COMPRESS_AND_RETRY:
            return await self._apply_compression(error, context, tool_name)

        elif strategy == RecoveryStrategy.SIMPLIFY_TOOLS:
            return self._apply_tool_simplification(error, tool_name)

        # Shouldn't reach here, but be safe
        return RecoveryAction(
            strategy_used=strategy,
            should_continue=False,
            attempt_number=self._attempt_count,
        )

    def _apply_guidance(self, error: Exception, tool_name: Optional[str]) -> RecoveryAction:
        """Add error context as guidance for the LLM."""
        error_msg = str(error)[:300]

        if tool_name:
            guidance = (
                f"The previous attempt to use '{tool_name}' failed with: {error_msg}. "
                f"Please try a different approach or use a different tool."
            )
        else:
            guidance = (
                f"The previous attempt failed with: {error_msg}. "
                f"Please try a different approach."
            )

        return RecoveryAction(
            strategy_used=RecoveryStrategy.RETRY_WITH_GUIDANCE,
            should_continue=True,
            guidance_message=guidance,
            attempt_number=self._attempt_count,
        )

    async def _apply_compression(
        self, error: Exception, context, tool_name: Optional[str]
    ) -> RecoveryAction:
        """Compress context and retry."""
        overflow = is_context_overflow_error(error)
        compressed = False
        if self.compress_fn is not None:
            try:
                import inspect

                if "overflow" in inspect.signature(self.compress_fn).parameters:
                    compressed = bool(await self.compress_fn(context, overflow=overflow))
                else:
                    compressed = bool(await self.compress_fn(context))
            except Exception as compress_error:
                logger.warning(f"Recovery compression failed: {compress_error}")
        elif self.context_compressor is not None and hasattr(context, "messages"):
            if hasattr(self.context_compressor, "compress_if_needed"):
                try:
                    result = await self.context_compressor.compress_if_needed(
                        context.messages, preserve_recent=6
                    )
                    if result.was_compressed:
                        compressed = True
                        context.messages = result.messages
                        logger.info(
                            f"Recovery compressed context: {result.original_count} -> "
                            f"{result.compressed_count} messages"
                        )
                except Exception as compress_error:
                    logger.warning(f"Recovery compression failed: {compress_error}")
            else:
                # A ContextEngine landed here. It cannot compress from messages
                # alone (it sizes the full request shape, which recovery does
                # not have), so without a wired compress_fn this strategy is a
                # no-op. Say so loudly — this exact mismatch previously hid
                # behind a blanket `except Exception` for months and silently
                # disabled the COMPRESS_AND_RETRY leg in the default config.
                logger.error(
                    "COMPRESS_AND_RETRY cannot run: context_compressor %s has no "
                    "compress_if_needed and no compress_fn was wired. Recovery "
                    "will retry without compaction.",
                    type(self.context_compressor).__name__,
                )

        # Only claim a refresh actually happened when it did — telling the
        # model the context changed when nothing did invites it to repeat the
        # exact request that just failed.
        error_msg = str(error)[:200]
        if compressed:
            guidance = (
                f"Context has been refreshed. Previous error: {error_msg}. "
                f"Please try again with a fresh approach."
            )
        else:
            guidance = (
                f"Previous error: {error_msg}. "
                f"Please try a different, more focused approach."
            )

        return RecoveryAction(
            strategy_used=RecoveryStrategy.COMPRESS_AND_RETRY,
            should_continue=True,
            guidance_message=guidance,
            attempt_number=self._attempt_count,
        )

    def _apply_tool_simplification(
        self, error: Exception, tool_name: Optional[str]
    ) -> RecoveryAction:
        """Exclude repeatedly failing tools."""
        excluded = set(self._excluded_tools)

        # Exclude EVERY tool that has hit the failure threshold, not just the
        # one that failed in the current step. The model can alternate between
        # similarly-shaped tools (e.g. meta_ads_insights ↔ google_ads_query
        # cross-confusion), so by the time SIMPLIFY_TOOLS fires, the earlier
        # offender may not be the current step's tool. Only exclusion across
        # all known problem tools breaks the loop.
        for name, count in self._tool_error_counts.items():
            if count >= 2 and name not in excluded:
                excluded.add(name)
                self._excluded_tools.add(name)
                logger.info(f"Excluding tool '{name}' after {count} failures")

        error_msg = str(error)[:200]
        if excluded:
            tools_str = ", ".join(sorted(excluded))
            guidance = (
                f"Previous error: {error_msg}. "
                f"The following tools have been excluded due to repeated failures: {tools_str}. "
                f"Please complete the task using the remaining available tools."
            )
        else:
            guidance = (
                f"Previous error: {error_msg}. "
                f"Please try a completely different approach to solve this task."
            )

        return RecoveryAction(
            strategy_used=RecoveryStrategy.SIMPLIFY_TOOLS,
            should_continue=True,
            guidance_message=guidance,
            excluded_tools=excluded if excluded else None,
            attempt_number=self._attempt_count,
        )

    @property
    def excluded_tools(self) -> Set[str]:
        """Get the set of currently excluded tools."""
        return set(self._excluded_tools)
