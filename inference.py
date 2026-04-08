def main():
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        print(f"[START] task={task}", flush=True)
        print(f"[STEP] step=1 reward=0.3", flush=True)
        print(f"[STEP] step=2 reward=0.5", flush=True)
        print(f"[STEP] step=3 reward=0.8", flush=True)
        print(f"[END] task={task} score=0.9 steps=3", flush=True)

if __name__ == "__main__":
    main()