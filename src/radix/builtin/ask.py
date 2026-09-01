from __future__ import annotations

from typing import Callable

from ..tool import tool

# Built-in interactive tool: ask_question presents a question with numbered
# choices to the user and returns their selection (or custom text).


@tool
def ask_question(question: str, choices: list[str], input_fn: Callable[[str], str] | None = None) -> str:
    """Ask the user a question with selectable choices.

    Displays the question and a numbered list of choices. The user may
    answer by typing the number of the choice they want, or by typing
    their own custom text.

    Args:
        question: The question to ask the user.
        choices: The list of selectable choices to present.
        input_fn: Optional callable that takes a prompt string and
            returns user input. When provided, it is used instead of
            ``input()`` (useful for testing).

    Returns:
        The user's answer as a string — either the text of the selected
        choice, or their custom text if they typed something else.
    """
    if input_fn is None:
        input_fn = input
    print(f"\n{question}")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")
    answer = input_fn("Your answer (number or custom text): ").strip()
    if answer.isdigit():
        index = int(answer) - 1
        if 0 <= index < len(choices):
            return choices[index]
        return (
            f"Error: {answer} is not a valid choice number. "
            f"Please pick a number between 1 and {len(choices)}."
        )
    return answer
