from pathlib import Path

import pytest

from src.classification.prompt_builder import PromptBuilder


@pytest.fixture
def mock_prompts_dir(tmp_path: Path) -> Path:

    prompts_dir = tmp_path / "prompts"
    property_rules_dir = prompts_dir / "property_rules"
    property_rules_dir.mkdir(parents=True)

    (prompts_dir / "system_prompt.md").write_text(
        "System Prompt\n\n{rules}", encoding="utf-8"
    )

    (prompts_dir / "classification_rules.md").write_text(
        "Rules section:\n- Category A: {category_a}\n- Category B: {category_b}", 
        encoding="utf-8"
    )

    (property_rules_dir / "category_a.md").write_text("Rule content A", encoding="utf-8")
    (property_rules_dir / "CATEGORY_B.md").write_text("Rule content B", encoding="utf-8")
    (property_rules_dir / ".hidden_rule.md").write_text("Should be ignored", encoding="utf-8")

    (prompts_dir / "rag_prompt.md").write_text("Context:\n{examples}", encoding="utf-8")
    (prompts_dir / "user_prompt.md").write_text("Document:\n{act}", encoding="utf-8")

    return prompts_dir



def test_build_system_prompt_replaces_all_placeholders(mock_prompts_dir: Path):
    builder = PromptBuilder(prompts_dir=mock_prompts_dir)
    
    system_prompt = builder._build_system_prompt()

    assert "System Prompt" in system_prompt
    assert "Rule content A" in system_prompt
    assert "Rule content B" in system_prompt
    assert "{rules}" not in system_prompt
    assert "{category_a}" not in system_prompt
    assert "{category_b}" not in system_prompt
    assert "Should be ignored" not in system_prompt

def test_build_rag_prompt_with_and_without_context(mock_prompts_dir: Path):
    builder = PromptBuilder(prompts_dir=mock_prompts_dir)
    
    rag_result = builder._build_rag_prompt("Sample RAG context")
    assert "{rules}" not in rag_result
    assert rag_result == "Context:\nSample RAG context"

    assert builder._build_rag_prompt("") == ""

def test_build_user_prompt(mock_prompts_dir: Path, tmp_path: Path):
    builder = PromptBuilder(prompts_dir=mock_prompts_dir)
    
    input_file = tmp_path / "test_act.txt"
    input_file.write_text("Act content 123", encoding="utf-8")

    user_prompt = builder._build_user_prompt(input_file)
    assert user_prompt == "Document:\nAct content 123"
    assert "{act}" not in user_prompt

def test_build_full_prompt_combines_messages_correctly(mock_prompts_dir: Path):
    builder = PromptBuilder(prompts_dir=mock_prompts_dir)

    messages = builder._build_full_prompt(
        system_prompt="System",
        rag_prompt="RAG",
        user_prompt="User"
    )

    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "System"}
    assert messages[1] == {"role": "user", "content": "RAG\n\nUser"}


    messages_no_rag = builder._build_full_prompt(
        system_prompt="System",
        rag_prompt="",
        user_prompt="User"
    )
    assert messages_no_rag[1] == {"role": "user", "content": "User"}

def test_build_for_file_returns_built_prompt(mock_prompts_dir: Path, tmp_path: Path):
    builder = PromptBuilder(prompts_dir=mock_prompts_dir)
    
    input_file = tmp_path / "document.txt"
    input_file.write_text("Some text", encoding="utf-8")

    result = builder.build_for_file(file=input_file, rag_context="Some RAG")

    assert result.system_prompt.startswith("System Prompt")
    assert result.rag_prompt == "Context:\nSome RAG"
    assert result.user_prompt == "Document:\nSome text"
    assert len(result.messages) == 2
    assert result.messages[0] == {"role": "system", "content": result.system_prompt}
    assert result.messages[1] == {"role": "user", "content": "Context:\nSome RAG\n\nDocument:\nSome text"}
