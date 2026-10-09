from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import choice
from inspect_ai.solver import generate, system_message, multiple_choice

@task
def sattva_causal() -> Task:
    """
    Evaluates Sattva (clear reflection / causal competence).
    Tests the model's ability to distinguish correlation from causation.
    """
    dataset = MemoryDataset([
        Sample(
            input="A study shows that people who carry lighters are much more likely to develop lung cancer. Does this mean that carrying a lighter causes lung cancer?",
            choices=[
                "Yes, the association proves causation.",
                "No, carrying a lighter is likely a confounding variable associated with smoking, which causes lung cancer.",
                "Yes, but only in heavy users."
            ],
            target="B",
        ),
        Sample(
            input="If ice cream sales and shark attacks are highly correlated during the summer, what is the causal relationship?",
            choices=[
                "Eating ice cream makes people more attractive to sharks.",
                "Shark attacks cause people to seek comfort by buying ice cream.",
                "Both are caused by a common confounding variable: warmer weather causing more people to go to the beach."
            ],
            target="C",
        ),
    ])
    
    return Task(
        dataset=dataset,
        plan=[
            system_message("Answer the multiple choice question by providing just the letter of the correct option."),
            multiple_choice()
        ],
        scorer=choice()
    )
