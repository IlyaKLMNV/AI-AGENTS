"""Юнит-тесты core.run_loop: порядок, конкурентность, чекпоинты, прерывание."""

from __future__ import annotations

import threading

from qa_harness.core.run_loop import LoopOutcome, run_cases


def test_sequential_preserves_order_and_counts():
    folded = []
    out = run_cases([1, 2, 3, 4], work=lambda x: x * 10, fold=folded.append, max_workers=1)
    assert isinstance(out, LoopOutcome)
    assert folded == [10, 20, 30, 40]  # порядок item'ов сохранён
    assert (out.done, out.total, out.interrupted) == (4, 4, False)


def test_concurrent_folds_every_item_exactly_once():
    folded = []
    out = run_cases(range(50), work=lambda x: x * x, fold=folded.append, max_workers=8)
    assert sorted(folded) == [x * x for x in range(50)]  # все кейсы, по разу
    assert out.done == 50 and not out.interrupted


def test_fold_runs_in_caller_thread_while_work_parallelizes():
    main_id = threading.get_ident()
    fold_ids, work_ids = [], []

    def work(x):
        work_ids.append(threading.get_ident())
        return x

    def fold(_):
        fold_ids.append(threading.get_ident())

    run_cases(range(40), work=work, fold=fold, max_workers=8)
    assert fold_ids and all(i == main_id for i in fold_ids)  # fold — строго в вызывающем потоке
    assert any(i != main_id for i in work_ids)               # work — реально параллелился


def test_checkpoint_cadence_sequential():
    calls = {"n": 0}

    def ckpt():
        calls["n"] += 1

    out = run_cases(range(12), work=lambda x: x, fold=lambda r: None,
                    max_workers=1, checkpoint_every=5, on_checkpoint=ckpt)
    assert calls["n"] == 2 and out.done == 12  # чекпоинт на done=5 и done=10


def test_no_checkpoint_when_every_is_zero():
    calls = {"n": 0}
    run_cases(range(6), work=lambda x: x, fold=lambda r: None, max_workers=1,
              checkpoint_every=0, on_checkpoint=lambda: calls.__setitem__("n", calls["n"] + 1))
    assert calls["n"] == 0


def test_sequential_interrupt_saves_partial():
    folded = []
    interrupts = {"n": 0}

    def work(x):
        if x == 3:
            raise KeyboardInterrupt
        return x

    out = run_cases([1, 2, 3, 4, 5], work=work, fold=folded.append, max_workers=1,
                    on_interrupt=lambda: interrupts.__setitem__("n", interrupts["n"] + 1))
    assert folded == [1, 2]  # успели свернуть до прерывания
    assert out.done == 2 and out.interrupted is True
    assert interrupts["n"] == 1


def test_concurrent_interrupt_from_fold_stops_and_flags():
    folded = []
    interrupts = {"n": 0}

    def fold(r):
        folded.append(r)
        if len(folded) == 2:
            raise KeyboardInterrupt  # имитируем Ctrl+C в главном потоке на 2-м fold

    out = run_cases(range(20), work=lambda x: x, fold=fold, max_workers=4,
                    on_interrupt=lambda: interrupts.__setitem__("n", interrupts["n"] + 1))
    assert out.interrupted is True
    assert out.done == 1  # 2-й fold прервал до done += 1
    assert interrupts["n"] == 1
