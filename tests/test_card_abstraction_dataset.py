from src.card_abstraction.canonical import canonicalize
from src.card_abstraction.domain import Street


def test_sample_unique_situations_is_seed_reproducible_and_deduplicated():
    from src.card_abstraction.dataset import sample_unique_situations

    first = sample_unique_situations(Street.FLOP, 100, 20260927)
    second = sample_unique_situations(Street.FLOP, 100, 20260927)

    assert [canonicalize(item) for item in first] == [canonicalize(item) for item in second]
    assert len({canonicalize(item) for item in first}) == 100


def test_sample_unique_situations_rejects_non_positive_count():
    from src.card_abstraction.dataset import sample_unique_situations

    try:
        sample_unique_situations(Street.TURN, 0, 20260927)
    except ValueError as error:
        assert "count" in str(error)
    else:
        raise AssertionError("Нулевой размер выборки должен быть отклонён")
