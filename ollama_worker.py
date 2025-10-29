import argparse
import json
import sys

try:
    import ollama  # type: ignore
except Exception as e:
    print(json.dumps({
        "ok": False,
        "error": f"Failed to import ollama: {str(e)}"
    }))
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama worker bridge")
    parser.add_argument("--check", action="store_true", help="Only verify availability")
    parser.add_argument("--model", type=str, default="llama3.1:8b")
    parser.add_argument("--mode", type=str, choices=["chat", "summary"], default="chat")
    parser.add_argument("--system", type=str, default="")
    parser.add_argument("--user", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_predict", type=int, default=500)

    args = parser.parse_args()

    if args.check:
        try:
            ollama.list()
            print(json.dumps({"ok": True}))
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}))
        return

    try:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        if args.user:
            messages.append({"role": "user", "content": args.user})

        response = ollama.chat(
            model=args.model,
            messages=messages,
            options={
                "temperature": args.temperature,
                "num_predict": args.num_predict,
            },
        )

        content = response.get("message", {}).get("content", "")
        print(json.dumps({
            "ok": True,
            "answer": content
        }))
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": str(e)
        }))


if __name__ == "__main__":
    main()


