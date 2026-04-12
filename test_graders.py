import itertools
from tasks.easy import grade as easy_grade
from tasks.medium import grade as medium_grade
from tasks.hard import grade as hard_grade

def test_grader():
    schemes = ["PM Ujjwala Yojana", "Ayushman Bharat", "Wrong Scheme", ""]
    questions_combos = [
        [],
        ["ask_occupation"],
        ["ask_occupation", "ask_bpl", "ask_gender"],
        ["ask_location", "ask_gender", "ask_occupation", "ask_bpl"],
        ["ask_location"] * 10
    ]
    steps_taken = [0, 4, 6, 8, 10, 15]
    total_rewards = [0.0, 0.5, 1.0, 10.0, -5.0]

    for g_name, grader in [("easy", easy_grade), ("medium", medium_grade), ("hard", hard_grade)]:
        for s in schemes:
            for qs in questions_combos:
                for st in steps_taken:
                    for tr in total_rewards:
                        res = grader(s, qs, st, tr)
                        score = res["score"]
                        if score <= 0.0 or score >= 1.0:
                            print(f"FAILED {g_name} -> {score}")
                            return
    print("ALL COMBOS SECURE 100%")

if __name__ == "__main__":
    test_grader()
