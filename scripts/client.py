#!/usr/bin/env python3
"""Client script to talk to the brainsurgery serving server."""

from __future__ import annotations

import json
import sys
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}") as resp:
        return json.loads(resp.read())


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def stream_post(path: str, body: dict) -> None:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            choice = chunk["choices"][0]
            if "text" in choice:
                print(choice["text"], end="", flush=True)
            elif "delta" in choice and "content" in choice["delta"]:
                print(choice["delta"]["content"], end="", flush=True)
    print()


def main():
    print("=== Health ===")
    print(json.dumps(get("/health"), indent=2))

    print("\n=== Models ===")
    print(json.dumps(get("/v1/models"), indent=2))

    if "--chat" in sys.argv:
        messages: list[dict] = []
        print("\n=== Interactive Chat (type 'quit' to exit) ===\n")
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break
            messages.append({"role": "user", "content": user_input})
            print("assistant> ", end="", flush=True)
            collected = ""
            data = json.dumps({
                "model": "default",
                "messages": messages,
                "max_tokens": 128,
                "temperature": 0.8,
                "stream": True,
            }).encode()
            req = urllib.request.Request(
                f"{BASE_URL}/v1/chat/completions", data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        print(text, end="", flush=True)
                        collected += text
            print()
            messages.append({"role": "assistant", "content": collected})
    else:
        print("\n=== Completion (non-streaming) ===")
        result = post("/v1/completions", {
            "model": "default",
            "prompt": "The future of AI is",
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": False,
        })
        print(json.dumps(result, indent=2))

        print("\n=== Completion (streaming) ===")
        stream_post("/v1/completions", {
            "model": "default",
            "prompt": "Once upon a time",
            "max_tokens": 64,
            "temperature": 0.8,
            "stream": True,
        })


if __name__ == "__main__":
    main()
