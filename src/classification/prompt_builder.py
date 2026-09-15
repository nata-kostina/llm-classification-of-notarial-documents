from pathlib import Path

from openai.types.chat import ChatCompletionMessageParam

from src.classification.models import BuiltPrompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PromptBuilder:
    def __init__(self, prompts_dir: Path | None = None):
        self.prompts_dir = prompts_dir or (PROJECT_ROOT / "prompts")
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        system_template_path = self.prompts_dir / "system_prompt.md"
        system_template = system_template_path.read_text(encoding="utf-8")

        rules_path = self.prompts_dir / "classification_rules.md"
        rules_template = rules_path.read_text(encoding="utf-8")

        property_rules_dir = self.prompts_dir / "property_rules"

        files = [item for item in property_rules_dir.glob("*.md") if not item.name.startswith(".")]

        for file in files:
            label = file.stem.lower()
            content = file.read_text(encoding="utf-8")
            rules_template = rules_template.replace(f"{{{label}}}", content)

        return system_template.replace("{rules}", rules_template)

    def _build_rag_prompt(self, rag_context: str) -> str:
        if not rag_context:
            return ""
        template_path = self.prompts_dir / "rag_prompt.md"
        rag_template = template_path.read_text(encoding="utf-8")

        return rag_template.replace("{examples}", rag_context)

    def _build_user_prompt(self, file: Path) -> str:
        user_template_path = self.prompts_dir / "user_prompt.md"
        user_template = user_template_path.read_text(encoding="utf-8")
        file_text = file.read_text(encoding="utf-8")

        return user_template.replace("{act}", file_text)

    def _build_full_prompt(self, *, system_prompt: str, rag_prompt: str, user_prompt: str) -> list[ChatCompletionMessageParam]:

        user_content_parts = [p for p in (rag_prompt, user_prompt) if p]
        user_content = "\n\n".join(user_content_parts)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_for_file(
        self,
        file: Path,
        rag_context: str,
    ) -> BuiltPrompt:
        rag_prompt = self._build_rag_prompt(rag_context)
        user_prompt = self._build_user_prompt(file)

        messages = self._build_full_prompt(system_prompt=self._system_prompt, rag_prompt=rag_prompt, user_prompt=user_prompt)

        return BuiltPrompt(
            system_prompt=self._system_prompt,
            rag_prompt=rag_prompt,
            user_prompt=user_prompt,
            messages=messages,
        )
