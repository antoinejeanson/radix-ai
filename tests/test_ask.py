from radix.builtin import ask as ask_module
from radix.builtin import ask_question


# Tests for the built-in ask_question tool.
#
# ask_question reads user input through the module-level
# `ask_module.input_fn` (deliberately kept out of the tool signature so it
# does not leak into the model-facing schema). Each test overrides it.
def _answers(values):
    it = iter(values)
    ask_module.input_fn = lambda _prompt: next(it)


def test_ask_question_select_by_number():
    _answers(["2"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana", "cherry"],
    )
    assert out == "banana"


def test_ask_question_select_first_by_number():
    _answers(["1"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana", "cherry"],
    )
    assert out == "apple"


def test_ask_question_select_last_by_number():
    _answers(["3"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana", "cherry"],
    )
    assert out == "cherry"


def test_ask_question_custom_text():
    _answers(["durian"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana"],
    )
    assert out == "durian"


def test_ask_question_invalid_number():
    _answers(["5"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana"],
    )
    assert out.startswith("Error")
    assert "not a valid choice number" in out


def test_ask_question_zero_number():
    _answers(["0"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana"],
    )
    assert out.startswith("Error")
    assert "not a valid choice number" in out


def test_ask_question_negative_number():
    _answers(["-1"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=["apple", "banana"],
    )
    # "-1" is not purely digits (has a minus sign), so it's treated as custom text
    assert out == "-1"


def test_ask_question_empty_choice_list():
    _answers(["1"])
    out = ask_question.run(
        question="Pick a fruit",
        choices=[],
    )
    assert out.startswith("Error")
    assert "not a valid choice number" in out


def test_ask_question_schema_has_no_input_fn():
    # Regression: the test-only input_fn must not leak into the model-facing
    # tool schema.
    props = ask_question.parameters["properties"]
    assert "input_fn" not in props
    assert set(props) == {"question", "choices"}
