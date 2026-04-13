import asyncio
import base64
import io
import itertools
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from openai import AsyncOpenAI


client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"  # замените на другую, если эта недоступна

GENERATE_CODE_PROMPT = """## ROLE
You are an expert ML engineer. Your task is to write a Python solution for a Kaggle-style competition.

## CONTEXT
### Competition instructions
{instructions}
{description_info}

### Available files (absolute paths on disk)
{files_info}

### Key paths
- Data directory: {data_dir}
- Output path: {workdir}/submission.csv

### Data preview
{train_info}
{sample_info}

## STEPS
1. Load train and test data from {data_dir}/
2. Preprocess: handle missing values with fillna or SimpleImputer
3. Train a model — prefer LightGBM or XGBoost, fall back to RandomForest if unavailable
4. Generate predictions on test data
5. Save predictions to {workdir}/submission.csv matching the exact format of sample_submission.csv

## CONSTRAINTS
- Use only absolute paths as listed above; never use /workdir or relative paths
- Available libraries: pandas, numpy, scikit-learn, lightgbm, xgboost

## OUTPUT
Output ONLY raw Python code — no markdown fences, no explanation, no comments outside the code.
"""

FIX_CODE_PROMPT = """## ROLE
You are an expert ML engineer debugging a broken Python script.

## CONTEXT
### Error output
{stderr}

### Standard output
{stdout}

### Original code
{code}

## STEPS
1. Identify the root cause of the error
2. Fix the issue while keeping the original logic intact
3. Ensure data is read from {workdir}/home/data/ and predictions are saved to {workdir}/submission.csv
4. Use only absolute paths; never use /workdir or relative paths

## OUTPUT
Output ONLY the fixed raw Python code — no markdown fences, no explanation.
"""


class Agent:
    def __init__(self):
        self._workdir: Path | None = None
        self._workdir_obj = None
        self._instructions: str = ""
        self._deps_installed: bool = False

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        text = get_message_text(message)
        print(f"=== run() called, message text preview: {text[:200]}...")
        if "validate" in text.lower() and self._workdir:
            await self._handle_validation(message, updater)
            return
        await self._handle_main_task(message, updater)

    async def _handle_validation(self, message: Message, updater: TaskUpdater) -> None:
        print("=== Validation branch")
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Validating submission..."),
        )
        submission_data = None
        for part in message.parts:
            if isinstance(part.root, FilePart):
                file_data = part.root.file
                if isinstance(file_data, FileWithBytes):
                    submission_data = base64.b64decode(file_data.bytes)
                    break
        if not submission_data:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: No submission file provided"))],
                name="validation_result",
            )
            return
        val_path = self._workdir / "validate_input.csv"
        val_path.write_bytes(submission_data)
        try:
            df = pd.read_csv(val_path)
            result = f"Valid submission: {len(df)} rows, columns: {list(df.columns)}"
        except Exception as e:
            result = f"Invalid submission: {e}"
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name="validation_result",
        )

    async def _handle_main_task(self, message: Message, updater: TaskUpdater) -> None:
        print("=== _handle_main_task started")
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Received task, extracting data..."),
        )

        instructions = ""
        tar_bytes = None

        for part in message.parts:
            if isinstance(part.root, TextPart):
                instructions += part.root.text + "\n"
            elif isinstance(part.root, FilePart):
                file_data = part.root.file
                if isinstance(file_data, FileWithBytes):
                    tar_bytes = base64.b64decode(file_data.bytes)

        if not tar_bytes:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message("No dataset tar.gz found in message"),
            )
            return

        self._workdir_obj = tempfile.TemporaryDirectory()
        self._workdir = Path(self._workdir_obj.name)
        self._instructions = instructions
        print(f"Created temporary workdir: {self._workdir}")

        try:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall(self._workdir)
            print(f"Extracted archive contents:")
            for root, dirs, files in os.walk(self._workdir):
                level = root.replace(str(self._workdir), '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
        except Exception as e:
            print(f"Extraction error: {e}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Failed to extract tar: {e}"),
            )
            return

        data_dir = self._workdir / "home" / "data"
        if not data_dir.exists():
            data_dir = self._workdir
            print(f"data_dir not found at {self._workdir}/home/data, using {data_dir} instead")
        else:
            print(f"data_dir = {data_dir}")

        files_list = [
            str(f.relative_to(self._workdir))
            for f in itertools.islice(
                (f for f in sorted(data_dir.rglob("*")) if f.is_file()), 30
            )
        ]
        files_info = "\n".join(files_list)
        print(f"files_info:\n{files_info}")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Files extracted:\n{files_info}\n\nGenerating solution..."),
        )

        description_info = self._read_first_file(data_dir, "description.md", 3000, "\nCOMPETITION DESCRIPTION:\n")
        sample_info = self._read_first_file(data_dir, "sample_submission*", 500, "\nSample submission:\n")
        train_info = self._read_first_file(data_dir, "train.csv", 1000, "\nTrain preview (first rows):\n")

        print("Generating code...")
        code = await self._generate_code(files_info, sample_info, train_info, data_dir, description_info)
        print(f"Generated code length: {len(code)}")
        print("=== Generated code ===\n", code, "\n=== End generated code ===")

        submission_bytes = await self._run_with_retry(code, updater)
        if submission_bytes is None:
            print("_run_with_retry returned None, exiting")
            return

        print("Solution generated successfully, sending artifact...")
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Solution done! Submitting..."),
        )

        await updater.add_artifact(
            parts=[Part(root=FilePart(
                file=FileWithBytes(
                    bytes=base64.b64encode(submission_bytes).decode("ascii"),
                    name="submission.csv",
                    mime_type="text/csv",
                )
            ))],
            name="submission",
        )
        print("Artifact sent")

    @staticmethod
    def _read_first_file(directory: Path, pattern: str, limit: int, prefix: str = "") -> str:
        for f in directory.rglob(pattern):
            try:
                return prefix + f.read_text()[:limit]
            except Exception:
                pass
        return ""

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        for marker in ["```python", "```"]:
            code = code.replace(marker, "")
        return code.strip()

    async def _generate_code(self, files_info: str, sample_info: str, train_info: str, data_dir: Path, description_info: str = "") -> str:
        prompt = GENERATE_CODE_PROMPT.format(
            instructions=self._instructions,
            description_info=description_info,
            files_info=files_info,
            data_dir=data_dir,
            workdir=self._workdir,
            train_info=train_info,
            sample_info=sample_info,
        )
        print("Sending prompt to LLM...")
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )
            print("LLM response received")
            return self._strip_code_fences(response.choices[0].message.content)
        except Exception as e:
            print(f"LLM error: {e}")
            raise

    async def _run_code(self, code: str) -> subprocess.CompletedProcess:
        script_path = self._workdir / "solution.py"
        script_path.write_text(code)
        print(f"Wrote solution.py to {script_path}")

        loop = asyncio.get_running_loop()

        if not self._deps_installed:
            print("Installing dependencies (scikit-learn, pandas, numpy)...")
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q",
                     "scikit-learn", "pandas", "numpy"],
                    capture_output=True,
                ),
            )
            self._deps_installed = True
            print("Dependencies installed")

        print(f"Running solution.py in {self._workdir}")
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "WORKDIR": str(self._workdir)},
            ),
        )
        print(f"Run finished with returncode={result.returncode}")
        if result.stdout:
            print(f"stdout: {result.stdout[:500]}")
        if result.stderr:
            print(f"stderr: {result.stderr[:500]}")
        return result

    async def _run_with_retry(self, code: str, updater: TaskUpdater) -> bytes | None:
        result = await self._run_code(code)
        submission_path = self._workdir / "submission.csv"

        if result.returncode == 0 and submission_path.exists():
            print("First attempt succeeded, submission file found")
            return submission_path.read_bytes()

        print(f"First attempt failed: returncode={result.returncode}, exists={submission_path.exists()}")
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("First attempt failed, fixing..."),
        )

        fix_prompt = FIX_CODE_PROMPT.format(
            stderr=result.stderr[-2000:],
            stdout=result.stdout[-500:],
            code=code,
            workdir=self._workdir,
        )
        print("Requesting LLM to fix the code...")
        fix_response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": fix_prompt}],
            max_tokens=3000,
        )
        fixed_code = self._strip_code_fences(fix_response.choices[0].message.content)
        print("Fixed code:\n", fixed_code)

        result2 = await self._run_code(fixed_code)

        if result2.returncode == 0 and submission_path.exists():
            print("Second attempt succeeded")
            return submission_path.read_bytes()

        print(f"Second attempt failed: returncode={result2.returncode}, exists={submission_path.exists()}")
        await updater.update_status(
            TaskState.failed,
            new_agent_text_message(f"Failed after retry.\nError: {result2.stderr[-1000:]}"),
        )
        return None