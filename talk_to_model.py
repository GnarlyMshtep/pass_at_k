#!/usr/bin/env python3
import argparse
import json
import sys
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def generate_response_vllm(client, messages, max_tokens=512, temperature=0.7):
    """Generate response using VLLM API"""
    response = client.chat.completions.create(
        model="Qwen2_5-7B",  # VLLM uses model name from serving
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def translate_if_needed(text, openai_api_key):
    """Use GPT-4o to detect and translate non-English content"""
    if not openai_api_key:
        return None

    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a translator. If the user's text contains any non-English content, translate it to English. If the text is already fully in English, respond with 'NO_TRANSLATION_NEEDED'. Otherwise, provide only the English translation without any preamble."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
        )
        translation = response.choices[0].message.content.strip()
        return None if translation == "NO_TRANSLATION_NEEDED" else translation
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Chat with VLLM-served models')
    parser.add_argument('--api-url', default="http://localhost:8000/v1", help='VLLM API base URL')
    parser.add_argument('--max-tokens', type=int, default=2048, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--system', default=None, help='System prompt to use for chat')
    parser.add_argument('--translate', action='store_true', default=False, help='Translate non-English responses using GPT-4o')

    args = parser.parse_args()

    # Initialize VLLM client
    vllm_client = OpenAI(
        api_key="EMPTY",  # VLLM doesn't require API key
        base_url=args.api_url,
    )

    # Get OpenAI API key for translation if needed
    openai_api_key = os.getenv("OPENAI_API_KEY") if args.translate else None
    if args.translate and not openai_api_key:
        print("Warning: --translate flag set but OPENAI_API_KEY not found in .env file")

    print(f"Connected to VLLM API at {args.api_url}")
    
    # Interactive chat mode
    print("\n=== CHAT MODE ===")
    print("Type 'quit' to exit, 'clear' to clear conversation\n")

    conversation = []

    # Add system prompt if provided
    if args.system:
        conversation.append({"role": "system", "content": args.system})
        print(f"System prompt set: {args.system[:50]}{'...' if len(args.system) > 50 else ''}\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'clear':
                conversation = []
                # Re-add system prompt if it was provided
                if args.system:
                    conversation.append({"role": "system", "content": args.system})
                print("Conversation cleared.\n")
                continue
            elif user_input == '':
                continue

            # Add user message
            conversation.append({"role": "user", "content": user_input})

            # Generate response
            response = generate_response_vllm(
                vllm_client,
                conversation,
                args.max_tokens,
                args.temperature
            )

            print(f"Assistant: {response}")

            # Translate if needed
            if args.translate and openai_api_key:
                translation = translate_if_needed(response, openai_api_key)
                if translation:
                    print(f"Translated: {translation}")

            print()  # Extra newline for readability

            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            # Remove the last user message if there was an error
            if conversation and conversation[-1]["role"] == "user":
                conversation.pop()

if __name__ == "__main__":
    main()