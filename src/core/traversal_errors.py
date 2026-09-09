from dataclasses import dataclass, field


@dataclass(frozen=True)
class TraversalFailureContext:
    iteration: int
    traversal_index: int | None
    traversing_player: int
    acting_player: int | None
    depth: int
    reason: str
    action_trace: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)


class TraversalFailure(RuntimeError):
    def __init__(self, context: TraversalFailureContext, cause: BaseException | None = None):
        self.context = context
        self.cause = cause
        super().__init__(context.reason)


def is_skippable_traversal_failure(error: BaseException) -> bool:
    return isinstance(error, TraversalFailure)
