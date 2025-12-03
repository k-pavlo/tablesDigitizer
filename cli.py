#!/usr/bin/env python3
"""
Simple Claude CLI - Perfect for Hackathons!

This is a beginner-friendly CLI tool to interact with Claude AI.
Everything is in one file to make it easy to understand and modify.

Author: Built with Claude for hackathon participants
"""

import os
import sys
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ClaudeCLI:
    """Simple wrapper for Claude API - easy to use and customize!"""

    def __init__(self):
        """Initialize the Claude API client"""
        # Get API key from environment variable
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            print("❌ Error: ANTHROPIC_API_KEY not found!")
            print("\n📝 To fix this:")
            print("1. Copy .env.example to .env")
            print("2. Add your API key from https://console.anthropic.com/")
            print("3. Run this script again")
            sys.exit(1)

        # Create the Claude client
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"  # Latest Claude model

    def ask(self, question, system_prompt=None):
        """
        Ask Claude a single question and get a response

        Args:
            question: Your question to Claude
            system_prompt: Optional instructions for how Claude should behave

        Returns:
            Claude's response as a string
        """
        try:
            print(f"\n🤔 Asking Claude: {question}\n")

            # Build the messages
            messages = [{"role": "user", "content": question}]

            # Call the API
            if system_prompt:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=messages
                )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=messages
                )

            # Extract and return the text
            answer = response.content[0].text
            return answer

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def chat(self):
        """
        Start an interactive chat session with Claude
        Type 'quit' or 'exit' to end the conversation
        """
        print("\n💬 Starting chat with Claude!")
        print("Type 'quit' or 'exit' to end the conversation\n")
        print("-" * 50)

        conversation_history = []

        while True:
            # Get user input
            user_input = input("\n😊 You: ").strip()

            # Check if user wants to quit
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye! Thanks for chatting!")
                break

            if not user_input:
                continue

            # Add user message to history
            conversation_history.append({
                "role": "user",
                "content": user_input
            })

            try:
                # Get Claude's response
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=conversation_history
                )

                # Extract Claude's reply
                claude_reply = response.content[0].text

                # Add Claude's response to history
                conversation_history.append({
                    "role": "assistant",
                    "content": claude_reply
                })

                # Display Claude's response
                print(f"\n🤖 Claude: {claude_reply}")

            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                break

    def analyze_file(self, filepath):
        """
        Send a file to Claude for analysis

        Args:
            filepath: Path to the file you want to analyze
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return f"❌ Error: File '{filepath}' not found!"

            # Read the file
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Ask Claude to analyze it
            question = f"Please analyze this file and summarize what it does:\n\n{content}"

            print(f"\n📄 Analyzing file: {filepath}\n")
            response = self.ask(question)
            return response

        except Exception as e:
            return f"❌ Error reading file: {str(e)}"


def print_help():
    """Display help information"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║          🚀 Simple Claude CLI - Help Guide                ║
╚═══════════════════════════════════════════════════════════╝

📖 Available Commands:

  python cli.py ask "your question here"
      Ask Claude a single question
      Example: python cli.py ask "What is Python?"

  python cli.py chat
      Start an interactive chat with Claude
      Example: python cli.py chat

  python cli.py analyze <filepath>
      Analyze a file with Claude
      Example: python cli.py analyze mycode.py

  python cli.py help
      Show this help message

╔═══════════════════════════════════════════════════════════╗
║  💡 Tips for Customization                                ║
╚═══════════════════════════════════════════════════════════╝

1. Change the model: Edit the 'self.model' line in __init__
2. Adjust response length: Change 'max_tokens' value
3. Add personality: Use system_prompt parameter in ask()
4. Build new commands: Add new methods to ClaudeCLI class

📚 API Documentation: https://docs.anthropic.com/
🔑 Get API Key: https://console.anthropic.com/
""")


def main():
    """Main entry point for the CLI"""

    # Check if user provided any arguments
    if len(sys.argv) < 2:
        print("\n❌ Error: No command provided")
        print_help()
        sys.exit(1)

    # Get the command
    command = sys.argv[1].lower()

    # Handle help command
    if command in ['help', '-h', '--help']:
        print_help()
        sys.exit(0)

    # Create CLI instance
    cli = ClaudeCLI()

    # Handle different commands
    if command == 'ask':
        # Check if question was provided
        if len(sys.argv) < 3:
            print("❌ Error: Please provide a question")
            print("Example: python cli.py ask 'What is Python?'")
            sys.exit(1)

        # Join all remaining arguments as the question
        question = ' '.join(sys.argv[2:])
        answer = cli.ask(question)
        print(f"\n🤖 Claude says:\n{answer}\n")

    elif command == 'chat':
        cli.chat()

    elif command == 'analyze':
        # Check if filepath was provided
        if len(sys.argv) < 3:
            print("❌ Error: Please provide a file path")
            print("Example: python cli.py analyze myfile.txt")
            sys.exit(1)

        filepath = sys.argv[2]
        result = cli.analyze_file(filepath)
        print(f"\n🤖 Analysis:\n{result}\n")

    else:
        print(f"\n❌ Error: Unknown command '{command}'")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
