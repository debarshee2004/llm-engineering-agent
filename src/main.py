import os
import sys
import json
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.style import Style

# Initialize Rich console
console = Console()

# --------------------------------------------------------------------------------
# 1. Configure LangChain client and load environment variables
# --------------------------------------------------------------------------------
load_dotenv()  # Load environment variables from .env file

class ModelSelector:
    """Handles model selection and configuration for different LLM providers"""

    # Define available models and their configurations
    AVAILABLE_MODELS = {
        "1": {
            "name": "OpenAI - GPT",
            "class": ChatOpenAI,
            "env_vars": ["OPENAI_API_KEY"],
            "default_model": "gpt-3.5-turbo"
        },
        "2": {
            "name": "Google - Gemini",
            "class": ChatGoogleGenerativeAI,
            "env_vars": ["GOOGLE_API_KEY"],
            "default_model": "gemini-pro"
        },
        "3": {
            "name": "xAI - Grok Beta",
            "class": ChatXAI,
            "env_vars": ["XAI_API_KEY"],
            "default_model": "grok-beta"
        },
        "4": {
            "name": "Anthropic - Claude",
            "class": ChatAnthropic,
            "env_vars": ["ANTHROPIC_API_KEY"],
            "default_model": "claude-3-sonnet-20240229"
        },
        "5": {
            "name": "Local - Ollama",
            "class": OllamaLLM,
            "env_vars": [],
            "default_model": "llama2"
        }
    }

    @classmethod
    def display_available_models(cls) -> None:
        """Display available models in a formatted table"""
        table = Table(title="Available Language Models")
        table.add_column("Choice", style="cyan")
        table.add_column("Provider - Model", style="green")
        table.add_column("Required API Key", style="yellow")

        for key, model in cls.AVAILABLE_MODELS.items():
            api_keys = ", ".join(model["env_vars"]) if model["env_vars"] else "None"
            table.add_row(key, model["name"], api_keys)

        console.print(table)

    @classmethod
    def get_model_choice(cls) -> str:
        """Get user's model choice with input validation"""
        while True:
            cls.display_available_models()
            choice = console.input("\n[bold cyan]Enter your choice (1-5):[/bold cyan] ").strip()

            if choice in cls.AVAILABLE_MODELS:
                return choice

            console.print("[red]Invalid choice. Please try again.[/red]")

    @classmethod
    def create_chat_model(cls, choice: str = None):
        """Create and return the appropriate chat model instance"""
        if choice is None:
            choice = cls.get_model_choice()

        model_config = cls.AVAILABLE_MODELS[choice]
        model_class = model_config["class"]

        # Verify required environment variables
        missing_vars = [var for var in model_config["env_vars"]
                        if not os.getenv(var)]

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables for {model_config['name']}: "
                f"{', '.join(missing_vars)}"
            )

        # Base configuration for all models
        config = {
            "model": model_config["default_model"],
            "streaming": True
        }

        # Add model-specific configurations
        if choice == "1":  # OpenAI
            config.update({
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_tokens": 8000
            })
        elif choice == "2":  # Google
            config.update({
                "google_api_key": os.getenv("GOOGLE_API_KEY"),
                "max_output_tokens": 8000
            })
        elif choice == "3":  # xAI
            config.update({
                "xai_api_key": os.getenv("XAI_API_KEY"),
                "max_tokens": 8000
            })
        elif choice == "4":  # Anthropic
            config.update({
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "max_tokens": 8000
            })
        elif choice == "5":  # Ollama
            config.update({
                "model": "llama2"  # Can be changed to any locally available model
            })

        return model_class(**config)

try:
    # Get chat model instance
    chat_model = ModelSelector.create_chat_model()
    console.print(f"\n[green]Successfully initialized chat model![/green]")

except ValueError as e:
    console.print(f"\n[red]Error:[/red] {str(e)}")
except Exception as e:
    console.print(f"\n[red]Unexpected error:[/red] {str(e)}")

# --------------------------------------------------------------------------------
# 2. Define our schema using Pydantic for type safety
# --------------------------------------------------------------------------------
class FileToCreate(BaseModel):
    path: str
    content: str

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str

class AssistantResponse(BaseModel):
    assistant_reply: str
    files_to_create: Optional[List[FileToCreate]] = None
    files_to_edit: Optional[List[FileToEdit]] = None

# --------------------------------------------------------------------------------
# 3. system prompt
# --------------------------------------------------------------------------------
system_prompt = dedent("""\
    You are a senior software engineer called Agent Engineer with decades of experience across all programming domains.
    Your expertise spans writing production standard code, system design, algorithms, testing, and best practices.
    You provide thoughtful, well-structured solutions while explaining your reasoning.

    Core capabilities:
    1. Code Analysis & Discussion
       - Analyze code with expert-level insight
       - Explain complex concepts clearly
       - Suggest optimizations and best practices
       - Debug issues with precision

    2. File Operations:
       a) Read existing files
          - Access user-provided file contents for context
          - Analyze multiple files to understand project structure

       b) Create new files
          - Generate complete new files with proper structure
          - Create complementary files (tests, configs, etc.)

       c) Edit existing files
          - Make precise changes using diff-based editing
          - Modify specific sections while preserving context
          - Suggest refactoring improvements

    Output Format:
    You must provide responses in this JSON structure:
    {
      "assistant_reply": "Your main explanation or response",
      "files_to_create": [
        {
          "path": "path/to/new/file",
          "content": "complete file content"
        }
      ],
      "files_to_edit": [
        {
          "path": "path/to/existing/file",
          "original_snippet": "exact code to be replaced",
          "new_snippet": "new code to insert"
        }
      ]
    }

    Guidelines:
    1. For normal responses, use 'assistant_reply'
    2. When creating files, include full content in 'files_to_create'
    3. For editing files:
       - Use 'files_to_edit' for precise changes
       - Include enough context in original_snippet to locate the change
       - Ensure new_snippet maintains proper indentation
       - Prefer targeted edits over full file replacements
    4. Always explain your changes and reasoning
    5. Consider edge cases and potential impacts
    6. Follow language-specific best practices
    7. Suggest tests or validation steps when appropriate

    Remember: You're a senior engineer - be thorough, precise, and thoughtful in your solutions.
    """)

# --------------------------------------------------------------------------------
# 4. Helper functions
# --------------------------------------------------------------------------------

def read_local_file(file_path: str) -> str:
    """Return the text content of a local file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_file(path: str, content: str):
    """Create (or overwrite) a file at 'path' with the given 'content'."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # ensures any dirs exist
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[green]✓[/green] Created/updated file at '[cyan]{file_path}[/cyan]'")

    # Record the action
    conversation_history.append(
        AIMessage(content=f"✓ Created/updated file at '{file_path}'")
    )

    # Add the actual content to conversation context
    normalized_path = normalize_path(str(file_path))
    conversation_history.append(
        SystemMessage(content=f"Content of file '{normalized_path}':\n\n{content}")
    )

def show_diff_table(files_to_edit: List[FileToEdit]) -> None:
    """Show the user a table of proposed edits and confirm"""
    if not files_to_edit:
        return

    table = Table(title="Proposed Edits", show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("File Path", style="cyan")
    table.add_column("Original", style="red")
    table.add_column("New", style="green")

    for edit in files_to_edit:
        table.add_row(edit.path, edit.original_snippet, edit.new_snippet)

    console.print(table)

def apply_diff_edit(path: str, original_snippet: str, new_snippet: str):
    """Reads the file at 'path', replaces the first occurrence of 'original_snippet' with 'new_snippet', then overwrites."""
    try:
        content = read_local_file(path)
        if original_snippet in content:
            updated_content = content.replace(original_snippet, new_snippet, 1)
            create_file(path, updated_content)  # This will now also update conversation context
            console.print(f"[green]✓[/green] Applied diff edit to '[cyan]{path}[/cyan]'")
            conversation_history.append(
                AIMessage(content=f"✓ Applied diff edit to '{path}'")
            )
        else:
            # Add debug info about the mismatch
            console.print(f"[yellow]⚠[/yellow] Original snippet not found in '[cyan]{path}[/cyan]'. No changes made.", style="yellow")
            console.print("\nExpected snippet:", style="yellow")
            console.print(Panel(original_snippet, title="Expected", border_style="yellow"))
            console.print("\nActual file content:", style="yellow")
            console.print(Panel(content, title="Actual", border_style="yellow"))
    except FileNotFoundError:
        console.print(f"[red]✗[/red] File not found for diff editing: '[cyan]{path}[/cyan]'", style="red")

def try_handle_add_command(user_input: str) -> bool:
    """
    If user_input starts with '/add', read that file and insert its content
    into conversation as a system message. Returns True if handled; else False.
    """
    prefix = "/add "
    if user_input.strip().lower().startswith(prefix):
        file_path = user_input[len(prefix):].strip()
        try:
            content = read_local_file(file_path)
            conversation_history.append(
                SystemMessage(content=f"Content of file '{file_path}':\n\n{content}")
            )
            console.print(f"[green]✓[/green] Added file '[cyan]{file_path}[/cyan]' to conversation.\n")
        except OSError as err:
            console.print(f"[red]✗[/red] Could not add file '[cyan]{file_path}[/cyan]': {err}\n", style="red")
        return True
    return False

def ensure_file_in_context(file_path: str) -> bool:
    """
    Ensures the file content is in the conversation context.
    Returns True if successful, False if file not found.
    """
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        file_marker = f"Content of file '{normalized_path}'"
        if not any(file_marker in msg.content for msg in conversation_history):
            conversation_history.append(
                SystemMessage(content=f"{file_marker}:\n\n{content}")
            )
        return True
    except OSError:
        console.print(f"[red]✗[/red] Could not read file '[cyan]{file_path}[/cyan]' for editing context", style="red")
        return False

def normalize_path(path_str: str) -> str:
    """Return a canonical, absolute version of the path."""
    return str(Path(path_str).resolve())

# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------
conversation_history = [
    SystemMessage(content=system_prompt)
]

# --------------------------------------------------------------------------------
# 6. LangChain API interaction with streaming
# --------------------------------------------------------------------------------

def guess_files_in_message(user_message: str) -> List[str]:
    """
    Attempt to guess which files the user might be referencing.
    Returns normalized absolute paths.
    """
    recognized_extensions = [".css", ".html", ".js", ".py", ".json", ".md", ".java", ".go", "c", "cpp"]
    potential_paths = []
    for word in user_message.split():
        if any(ext in word for ext in recognized_extensions) or "/" in word:
            path = word.strip("',\"")
            try:
                normalized_path = normalize_path(path)
                potential_paths.append(normalized_path)
            except (OSError, ValueError):
                continue
    return potential_paths

def stream_langchain_response(user_message: str):
    """
    Streams the LLM chat completion response and handles structured output.
    Returns the final AssistantResponse.
    """
    potential_paths = guess_files_in_message(user_message)
    valid_files = {}

    # Try to read all potential files before the API call
    for path in potential_paths:
        try:
            content = read_local_file(path)
            valid_files[path] = content
            file_marker = f"Content of file '{path}'"
            if not any(file_marker in msg.content for msg in conversation_history):
                conversation_history.append(
                    SystemMessage(content=f"{file_marker}:\n\n{content}")
                )
        except OSError:
            error_msg = f"Cannot proceed: File '{path}' does not exist or is not accessible"
            console.print(f"[red]✗[/red] {error_msg}", style="red")
            continue

    conversation_history.append(HumanMessage(content=user_message))

    try:
        console.print("\nAssistant> ", style="bold blue", end="")
        full_content = ""

        # Stream the response using LangChain
        for chunk in chat_model.stream(conversation_history):
            if chunk.content:
                full_content += chunk.content
                console.print(chunk.content, end="")

        console.print()

        # Try to extract JSON from the response
        try:
            # First, try to find JSON-like structure in the response
            json_start = full_content.find('{')
            json_end = full_content.rfind('}') + 1

            if 0 <= json_start < json_end:
                json_content = full_content[json_start:json_end]
                parsed_response = json.loads(json_content)
            else:
                # If no JSON structure found, create a default response
                parsed_response = {
                    "assistant_reply": full_content,
                    "files_to_create": [],
                    "files_to_edit": []
                }

            # Ensure all required fields are present
            parsed_response.setdefault("assistant_reply", "")
            parsed_response.setdefault("files_to_create", [])
            parsed_response.setdefault("files_to_edit", [])

            # Handle file edits
            if parsed_response["files_to_edit"]:
                new_files_to_edit = []
                for edit in parsed_response["files_to_edit"]:
                    try:
                        edit_abs_path = normalize_path(edit["path"])
                        if edit_abs_path in valid_files or ensure_file_in_context(edit_abs_path):
                            edit["path"] = edit_abs_path
                            new_files_to_edit.append(edit)
                    except (OSError, ValueError):
                        console.print(f"[yellow]⚠[/yellow] Skipping invalid path: '{edit['path']}'", style="yellow")
                        continue
                parsed_response["files_to_edit"] = new_files_to_edit

            response_obj = AssistantResponse(**parsed_response)
            conversation_history.append(
                AIMessage(content=response_obj.assistant_reply)
            )

            return response_obj

        except json.JSONDecodeError as json_err:
            error_msg = f"Failed to parse JSON response: {str(json_err)}\nRaw content: {full_content[:200]}..."
            console.print(f"[red]✗[/red] {error_msg}", style="red")
            return AssistantResponse(
                assistant_reply=full_content,
                files_to_create=[],
                files_to_edit=[]
            )

    except Exception as err:
        error_msg = f"LLM API error: {str(err)}"
        console.print(f"\n[red]✗[/red] {error_msg}", style="red")
        return AssistantResponse(
            assistant_reply=error_msg,
            files_to_create=[],
            files_to_edit=[]
        )

# --------------------------------------------------------------------------------
# 7. Main interactive loop
# --------------------------------------------------------------------------------

def main():
    console.print(Panel.fit(
        "[bold blue]Welcome to LLM Engineering Agent with Structured Output[/bold blue] [green](and streaming)[/green]!🤖",
        border_style="blue"
    ))
    console.print(
        "To include a file in the conversation, use '[bold magenta]/add path/to/file[/bold magenta]'.\n"
        "Type '[bold red]exit[/bold red]' or '[bold red]quit[/bold red]' to end.\n"
    )

    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting.[/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break

        # If user is reading a file
        if try_handle_add_command(user_input):
            continue

        # Get streaming response from LangChain
        response_data = stream_langchain_response(user_input)

        # Create any files if requested
        if response_data.files_to_create:
            for file_info in response_data.files_to_create:
                create_file(file_info.path, file_info.content)

        # Show and confirm diff edits if requested
        if response_data.files_to_edit:
            show_diff_table(response_data.files_to_edit)
            confirm = console.input(
                "\nDo you want to apply these changes? ([green]y[/green]/[red]n[/red]): "
            ).strip().lower()
            if confirm == 'y':
                for edit_info in response_data.files_to_edit:
                    apply_diff_edit(edit_info.path, edit_info.original_snippet, edit_info.new_snippet)
            else:
                console.print("[yellow]ℹ[/yellow] Skipped applying diff edits.", style="yellow")

    console.print("[blue]Session finished.[/blue]")

if __name__ == "__main__":
    main()