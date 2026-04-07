import asyncio
import base64
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from openai import AsyncOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from messenger import Messenger

API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openai.bothub.ru/v1"
MODEL_ID = "gpt-4o-mini"

_llm = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

_MAX_TOKENS = 3000
_SCRIPT_TIMEOUT = 300
_MAX_FILES_LISTED = 30
_DESCRIPTION_PREVIEW = 3000
_SAMPLE_PREVIEW = 500
_TRAIN_PREVIEW = 1000
_ERR_TAIL = 2000
_STDOUT_TAIL = 500
_ERR_TAIL_SHORT = 1000


def _strip_fences(text: str) -> str:
    for fence in ("```python", "```"):
        text = text.replace(fence, "")
    return text.strip()


def _read_first_match(pattern: str, root: Path, max_chars: int, label: str) -> str:
    for path in root.rglob(pattern):
        try:
            return f"\n{label}:\n{path.read_text()[:max_chars]}"
        except Exception:
            pass
    return ""


class MLCompetitionAgent:
    def __init__(self) -> None:
        self.messenger = Messenger()
        self._workspace: Path | None = None
        self._tmp_dir = None
        self._task_description: str = ""

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        content = get_message_text(message)
        if "validate" in content.lower() and self._workspace is not None:
            await self._validate_submission(message, updater)
        else:
            await self._solve_competition(message, updater)

    # ------------------------------------------------------------------
    # Validation flow
    # ------------------------------------------------------------------

    async def _validate_submission(self, message: Message, updater: TaskUpdater) -> None:
        assert self._workspace is not None
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Checking submission file..."),
        )

        raw = self._extract_file_bytes(message)
        if raw is None:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: submission file is missing"))],
                name="validation_result",
            )
            return

        csv_path = self._workspace / "candidate_submission.csv"
        csv_path.write_bytes(raw)

        try:
            import pandas as pd
            frame = pd.read_csv(csv_path)
            outcome = f"OK — {len(frame)} rows, columns: {list(frame.columns)}"
        except Exception as exc:
            outcome = f"Rejected — {exc}"

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=outcome))],
            name="validation_result",
        )

    # ------------------------------------------------------------------
    # Main competition flow
    # ------------------------------------------------------------------

    async def _solve_competition(self, message: Message, updater: TaskUpdater) -> None:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Processing incoming task..."),
        )

        description, archive = self._unpack_message(message)

        if archive is None:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message("Expected a tar.gz dataset — none found"),
            )
            return

        self._tmp_dir = tempfile.TemporaryDirectory()
        self._workspace = Path(self._tmp_dir.name)
        self._task_description = description

        try:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                tar.extractall(self._workspace)
        except Exception as exc:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Could not unpack archive: {exc}"),
            )
            return

        data_root = self._workspace / "home" / "data"
        if not data_root.exists():
            data_root = self._workspace

        file_listing = self._build_file_listing(data_root)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Dataset ready:\n{file_listing}\n\nBuilding solution..."),
        )

        competition_desc = _read_first_match("description.md", data_root, _DESCRIPTION_PREVIEW, "COMPETITION DESCRIPTION")
        sample_csv = _read_first_match("sample_submission*", data_root, _SAMPLE_PREVIEW, "Sample submission format")
        train_csv = _read_first_match("train.csv", data_root, _TRAIN_PREVIEW, "Training data preview")

        script = await self._compose_solution(file_listing, sample_csv, train_csv, data_root, competition_desc)

        submission_bytes = await self._execute_with_retry(script, updater)
        if submission_bytes is None:
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Predictions generated, uploading..."),
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_file_bytes(message: Message) -> bytes | None:
        for part in message.parts:
            if isinstance(part.root, FilePart):
                f = part.root.file
                if isinstance(f, FileWithBytes):
                    return base64.b64decode(f.bytes)
        return None

    @staticmethod
    def _unpack_message(message: Message) -> tuple[str, bytes | None]:
        text_parts: list[str] = []
        archive: bytes | None = None
        for part in message.parts:
            if isinstance(part.root, TextPart):
                text_parts.append(part.root.text)
            elif isinstance(part.root, FilePart):
                f = part.root.file
                if isinstance(f, FileWithBytes):
                    archive = base64.b64decode(f.bytes)
        return "\n".join(text_parts), archive

    def _build_file_listing(self, data_root: Path) -> str:
        assert self._workspace is not None
        entries = [
            str(p.relative_to(self._workspace))
            for p in sorted(data_root.rglob("*"))[:_MAX_FILES_LISTED]
            if p.is_file()
        ]
        return "\n".join(entries)

    async def _compose_solution(
        self,
        file_listing: str,
        sample_csv: str,
        train_csv: str,
        data_root: Path,
        competition_desc: str = "",
    ) -> str:
        sections = [
            "You are solving a machine learning competition. Write a Python script that trains a model and produces a submission file.",
            f"## Competition task\n{self._task_description.strip()}",
        ]
        if competition_desc.strip():
            sections.append(competition_desc.strip())
        sections += [
            f"## File system layout\nAll paths are absolute.\n{file_listing}",
            f"## Paths\nInput data directory: {data_root}\nOutput file: {self._workspace}/submission.csv",
        ]
        if train_csv.strip():
            sections.append(train_csv.strip())
        if sample_csv.strip():
            sections.append(sample_csv.strip())
        sections.append(
            "## Requirements\n"
            f"1. Load data from `{data_root}/`.\n"
            "2. Train a model. Use LightGBM or XGBoost when possible; fall back to RandomForest.\n"
            "3. Impute or fill all missing values before fitting.\n"
            f"4. Save predictions to `{self._workspace}/submission.csv` using the exact columns from the sample submission.\n"
            "5. Only use: pandas, numpy, scikit-learn, lightgbm, xgboost.\n"
            "6. Use the absolute paths shown above — never `/workdir` or relative paths.\n"
            "7. Output raw Python only — no markdown fences, no prose."
        )
        prompt = "\n\n".join(sections)
        resp = await _llm.chat.completions.create(
            model=_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_MAX_TOKENS,
        )
        return _strip_fences(resp.choices[0].message.content)

    async def _run_script(self, code: str) -> subprocess.CompletedProcess:
        assert self._workspace is not None
        script_file = self._workspace / "solve.py"
        script_file.write_text(code)

        loop = asyncio.get_event_loop()

        await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "pandas", "numpy"],
                capture_output=True,
            ),
        )

        return await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, str(script_file)],
                capture_output=True,
                text=True,
                timeout=_SCRIPT_TIMEOUT,
                env={**os.environ, "WORKDIR": str(self._workspace)},
            ),
        )

    async def _execute_with_retry(self, code: str, updater: TaskUpdater) -> bytes | None:
        assert self._workspace is not None
        result = await self._run_script(code)
        output_csv = self._workspace / "submission.csv"

        if result.returncode == 0 and output_csv.exists():
            return output_csv.read_bytes()

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Initial run failed — attempting a fix..."),
        )

        fix_prompt = "\n\n".join([
            "A Python ML script failed at runtime. Fix it so it runs without errors.",
            f"## Stderr\n{result.stderr[-_ERR_TAIL:]}",
            f"## Stdout\n{result.stdout[-_STDOUT_TAIL:]}",
            f"## Script\n{code}",
            (
                f"Data directory: {self._workspace}/home/data/\n"
                f"Output file: {self._workspace}/submission.csv\n"
                "Rules: absolute paths only (no /workdir, no relative paths); "
                "return raw Python only, no markdown fences."
            ),
        ])
        fix_resp = await _llm.chat.completions.create(
            model=_MODEL_ID,
            messages=[{"role": "user", "content": fix_prompt}],
            max_tokens=_MAX_TOKENS,
        )
        fixed = _strip_fences(fix_resp.choices[0].message.content)

        result2 = await self._run_script(fixed)

        if result2.returncode == 0 and output_csv.exists():
            return output_csv.read_bytes()

        await updater.update_status(
            TaskState.failed,
            new_agent_text_message(f"Still failing after fix attempt.\n{result2.stderr[-_ERR_TAIL_SHORT:]}"),
        )
        return None
