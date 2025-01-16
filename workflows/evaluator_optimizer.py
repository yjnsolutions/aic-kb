import subprocess
from pathlib import Path
from typing import List, Literal, Optional

import litellm
import typer
import yaml
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from litellm import OpenAIError
from pydantic import BaseModel
from typing_extensions import Annotated

app = typer.Typer()


class EvaluationResult(BaseModel):
    success: bool
    feedback: Optional[str]


class EOConfig(BaseModel):
    prompt: str
    coder_model: str
    evaluator_model: Literal["openai/gpt-4o", "openai/o1-preview-2024-09-12"]
    max_iterations: int
    execution_command: List[str]
    context_editable: List[str]
    context_read_only: List[str]
    evaluator: Literal["default"]
    post_edit_commands: List[str] = []


class EvaluatorOptimizer:
    """
    Self Directed AI Coding Assistant
    """

    def __init__(self, config_path: Path):
        self.config = self.validate_config(config_path)

    @staticmethod
    def validate_config(config_path: Path) -> EOConfig:
        """Validate the yaml config file and return EOConfig object."""
        config_dict = yaml.safe_load(config_path.read_text())
        # If prompt ends with .md, read content from that file
        if config_dict["prompt"].endswith(".md"):
            prompt_path = Path(config_dict["prompt"])
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            with open(prompt_path) as f:
                config_dict["prompt"] = f.read()
        # Ensure 'execution_command' is a list of strings
        execution_command = config_dict.get("execution_command")
        if isinstance(execution_command, str):
            config_dict["execution_command"] = [execution_command]
        elif isinstance(execution_command, list):
            if not all(isinstance(cmd, str) for cmd in execution_command):
                raise ValueError("All items in 'execution_command' list must be strings.")
        else:
            raise ValueError("'execution_command' must be a string or a list of strings.")
        config = EOConfig(**config_dict)
        # Validate evaluator_model is one of the allowed values
        allowed_evaluator_models = {"openai/gpt-4o", "openai/o1-preview-2024-09-12"}
        if config.evaluator_model not in allowed_evaluator_models:
            raise ValueError(
                f"evaluator_model must be one of {allowed_evaluator_models}, " f"got {config.evaluator_model}"
            )
        # Validate we have at least 1 editable file
        if not config.context_editable:
            raise ValueError("At least one editable context file must be specified")
        # Validate all paths in context_editable and context_read_only exist
        for path in config.context_editable:
            if not Path(path).exists():
                raise FileNotFoundError(f"Editable context file not found: {path}")
        for path in config.context_read_only:
            if not Path(path).exists():
                raise FileNotFoundError(f"Read-only context file not found: {path}")
        # Map 'post-edit-commands' from YAML to 'post_edit_commands' in EOConfig
        post_edit_commands = config_dict.get("post-edit-commands", [])
        config_dict["post_edit_commands"] = post_edit_commands
        return config

    def parse_llm_json_response(self, str) -> str:
        """
        Parse and fix the response from an LLM that is expected to return JSON.
        """
        if "```" not in str:
            str = str.strip()
            self.file_log(f"raw pre-json-parse: {str}", print_message=False)
            return str
        # Remove opening backticks and language identifier
        str = str.split("```", 1)[-1].split("\n", 1)[-1]
        # Remove closing backticks
        str = str.rsplit("```", 1)[0]
        str = str.strip()
        self.file_log(f"post-json-parse: {str}", print_message=False)
        # Remove any leading or trailing whitespace
        return str

    def file_log(self, message: str, print_message: bool = True):
        if print_message:
            print(message)
        with open("logs/evaluator_optimizer_log.txt", "a+") as f:
            f.write(message + "\n")

    def create_new_ai_coding_prompt(
        self,
        iteration: int,
        base_input_prompt: str,
        execution_output: str,
        evaluation: EvaluationResult,
    ) -> str:
        if iteration == 0:
            return base_input_prompt
        else:
            return f"""
# Generate the next iteration of code to achieve the user's desired result based on their original instructions and the feedback from the previous attempt.
Generate a new prompt in the same style as the original instructions for the next iteration of code.
## This is your {iteration}th attempt to generate the code.
You have {self.config.max_iterations - iteration} attempts remaining.
## Here's the user's original instructions for generating the code:
{base_input_prompt}
## Here's the output of your previous attempt:
{execution_output}
## Here's feedback on your previous attempt:
{evaluation.feedback}"""

    def ai_code(self, prompt: str):
        model = Model(self.config.coder_model)
        coder = Coder.create(
            main_model=model,
            io=InputOutput(yes=True),
            fnames=self.config.context_editable,
            read_only_fnames=self.config.context_read_only,
            auto_commits=False,
            suggest_shell_commands=False,
            detect_urls=False,
        )
        coder.run(prompt)

    def execute(self) -> str:
        """Execute all commands and return the combined output as an XML-formatted string."""
        import xml.etree.ElementTree as ET

        results_element = ET.Element("results")
        for cmd in self.config.execution_command:
            command_element = ET.SubElement(results_element, "command")
            cmd_element = ET.SubElement(command_element, "cmd")
            cmd_element.text = cmd
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
                # Sanitize and wrap stdout in CDATA
                safe_stdout = result.stdout.replace("]]>", "]]]]><![CDATA[>")
                stdout_element = ET.SubElement(command_element, "stdout")
                stdout_element.text = f"<![CDATA[{safe_stdout}]]>"
                # Sanitize and wrap stderr in CDATA
                safe_stderr = result.stderr.replace("]]>", "]]]]><![CDATA[>")
                stderr_element = ET.SubElement(command_element, "stderr")
                stderr_element.text = f"<![CDATA[{safe_stderr}]]>"
            except Exception as e:
                # Sanitize and wrap error message in CDATA
                error_message = f"Exception: {str(e)}"
                safe_error_message = error_message.replace("]]>", "]]]]><![CDATA[>")
                stdout_element = ET.SubElement(command_element, "stdout")
                stdout_element.text = "<![CDATA[]]>"  # Empty CDATA section for stdout
                stderr_element = ET.SubElement(command_element, "stderr")
                stderr_element.text = f"<![CDATA[{safe_error_message}]]>"
            # Continue executing the next command regardless of success or failure
        # Convert the XML ElementTree to a string
        xml_output = ET.tostring(results_element, encoding="unicode")
        self.file_log(f"Execution output: \n{xml_output}", print_message=False)
        return xml_output

    def evaluate(self, execution_output: str) -> EvaluationResult:
        if self.config.evaluator != "default":
            raise ValueError(f"Custom evaluator {self.config.evaluator} not implemented")
        map_editable_fname_to_files = {
            Path(fname).name: Path(fname).read_text() for fname in self.config.context_editable
        }
        map_read_only_fname_to_files = {
            Path(fname).name: Path(fname).read_text() for fname in self.config.context_read_only
        }
        evaluation_prompt = f"""Evaluate this execution output and determine if it was successful based on the execution command, the user's desired result, the editable files, checklist, and the read-only files.
## Checklist:
Is the execution output reporting success or failure?
Did we miss any tasks? Review the User's Desired Result to see if we have satisfied all tasks.
Did we satisfy the user's desired result?
Ignore warnings
## User's Desired Result:
{self.config.prompt}
## Editable Files:
{map_editable_fname_to_files}
## Read-Only Files:
{map_read_only_fname_to_files}
## Execution Command:
{self.config.execution_command}
## Execution Output:
{execution_output}
## Response Format:
Be 100% sure to output JSON.parse compatible JSON.
That means no new lines.
Return a structured JSON response with the following structure: {{
    success: bool - true if the execution output generated by the execution command matches the Users Desired Result
    feedback: str | None - if unsuccessful, provide detailed feedback explaining what failed and how to fix it, or None if successful
}}"""
        self.file_log(
            f"Evaluation prompt: ({self.config.evaluator_model}):\n{evaluation_prompt}",
            print_message=False,
        )
        try:
            completion = litellm.completion(
                model=self.config.evaluator_model,
                messages=[
                    {
                        "role": "user",
                        "content": evaluation_prompt,
                    },
                ],
            )
            response_content = completion.choices[0].message.content
            self.file_log(
                f"Evaluation response: ({self.config.evaluator_model}):\n{response_content}",
                print_message=False,
            )
            evaluation = EvaluationResult.model_validate_json(self.parse_llm_json_response(response_content))
            return evaluation
        except OpenAIError as e:
            self.file_log(f"Error evaluating execution output for '{self.config.evaluator_model}'. Error: {e}.")
            raise ValueError(f"Failed to evaluate execution output: {e}") from e

    def execute_post_edit_commands(self):
        """Execute the post-edit commands specified in the configuration."""
        for command in self.config.post_edit_commands:
            self.file_log(f":computer: Executing command: {command}")
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True,
                    executable="/bin/bash",
                )
                self.file_log(f":memo: Command output:\n{result.stdout}")
                if result.stderr:
                    self.file_log(f":warning: Command error output:\n{result.stderr}")
            except subprocess.CalledProcessError as e:
                self.file_log(f":x: Command failed with error:\n{e}")

    def evaluate_optimize(self):
        evaluation = EvaluationResult(success=False, feedback=None)
        execution_output = ""
        success = False
        for i in range(self.config.max_iterations):
            self.file_log(f"\nIteration {i + 1}/{self.config.max_iterations}")
            self.file_log(":brain: Creating new prompt...")
            new_prompt = self.create_new_ai_coding_prompt(i, self.config.prompt, execution_output, evaluation)
            self.file_log(":robot_face: Generating AI code...")
            self.ai_code(new_prompt)
            self.file_log(":wrench: Running post-edit-commands...")
            self.execute_post_edit_commands()
            self.file_log(f":computer: Executing code... '{self.config.execution_command}'")
            execution_output = self.execute()
            self.file_log(f":mag: Evaluating results... '{self.config.evaluator_model}' + '{self.config.evaluator}'")
            evaluation = self.evaluate(execution_output)
            self.file_log(
                f":mag: Evaluation result: {':white_check_mark: Success' if evaluation.success else ':x: Failed'}"
            )
            if evaluation.feedback:
                self.file_log(f":speech_balloon: Feedback: \n{evaluation.feedback}")
            if evaluation.success:
                success = True
                self.file_log(f"\n:tada: Success achieved after {i + 1} iterations! Breaking out of iteration loop.")
                break
            else:
                self.file_log(
                    f"\n:arrows_counterclockwise: Continuing with next iteration... Have {self.config.max_iterations - i - 1} attempts remaining."
                )
        if not success:
            self.file_log("\n:no_entry_sign: Failed to achieve success within the maximum number of iterations.")
        self.file_log("\nDone.")


@app.command()
def evaluate_optimize(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
):
    """Run the AI Coding EvaluatorOptimizer with a config file"""
    eo = EvaluatorOptimizer(config)
    eo.evaluate_optimize()


def main():
    app()


if __name__ == "__main__":
    main()
