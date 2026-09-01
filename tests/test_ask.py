from radix.builtin import ask_question


# Tests for the built-in ask_question tool.
def test_ask_question_select_by_number():
    answers = iter(["2"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana", "cherry"],
        input_fn=lambda _: next(answers),
    )
    assert out == "banana"


def test_ask_question_select_first_by_number():
    answers = iter(["1"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana", "cherry"],
        input_fn=lambda _: next(answers),
    )
    assert out == "apple"


def test_ask_question_select_last_by_number():
    answers = iter(["3"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana", "cherry"],
        input_fn=lambda _: next(answers),
    )
    assert out == "cherry"


def test_ask_question_custom_text():
    answers = iter(["durian"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana"],
        input_fn=lambda _: next(answers),
    )
    assert out == "durian"


def test_ask_question_invalid_number():
    answers = iter(["5"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana"],
        input_fn=lambda _: next(answers),
    )
    assert out.startswith("Error")
    assert "not a valid choice number" in out


def test_ask_question_zero_number():
    answers = iter(["0"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana"],
        input_fn=lambda _: next(answers),
    )
    assert out.startswith("Error")
    assert "not a valid choice number" in out


def test_ask_question_negative_number():
    answers = iter(["-1"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana"],
        input_fn=lambda _: next(answers),
    )
    # "-1" is not purely digits (has a minus sign), so it's treated as custom text
    assert out == "-1"


def test_ask_question_empty_choice_list():
    answers = iter(["1"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=[],
        input_fn=lambda _: next(answers),
    )
    assert out.startswith("Error")
    assert "not a valid choice number" in out
