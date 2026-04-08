def run_task(task):
    step = 1
    total_reward = 0.0

    print(f"[TASK] {task.upper()}")
    print(f"[START] task={task}")

    reward = 0.5
    total_reward += reward
    print(f"[STEP] step={step} reward={reward}")
    step += 1

    if task == "easy":
        reward = 0.2
    elif task == "medium":
        reward = 0.3
    else:
        reward = 0.4

    total_reward += reward
    print(f"[STEP] step={step} reward={reward}")
    step += 1

    reward = 1.0
    total_reward += reward
    print(f"[STEP] step={step} reward={reward}")

    print(f"[END] task={task} score={round(total_reward,2)} steps={step}")


def main():
    run_task("easy")
    run_task("medium")
    run_task("hard")


if __name__ == "__main__":
    main()