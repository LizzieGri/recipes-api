import asyncio
import os
from typing import Any, List

import dotenv
from github import Auth, Github
from llama_index.core.agent.workflow import AgentOutput, ReActAgent, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

dotenv.load_dotenv()

model_name = (
    os.getenv("MODEL")
    or os.getenv("OPENAI_MODEL")
    or "gpt-6-luna"
).strip()
api_key = os.getenv("OPENAI_API_KEY")
api_base = (os.getenv("OPENAI_BASE_URL") or "").strip() or None

llm = OpenAI(
    model=model_name,
    api_key=api_key,
    api_base=api_base,
)

github_token = (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
git = Github(auth=Auth.Token(github_token)) if github_token else Github()

raw_repo_url = (
    os.getenv("GITHUB_REPO_URL")
    or os.getenv("REPO_URL")
    or "https://github.com/LizzieGri/recipes-api.git"
).strip().rstrip("/")

if raw_repo_url.endswith(".git"):
    raw_repo_url = raw_repo_url[:-4]

parts = raw_repo_url.split("/")
full_repo_name = f"{parts[-2]}/{parts[-1]}"
repo = git.get_repo(full_repo_name)


def get_pr_details(pr_num: int) -> dict[str, Any]:
    """Returns pull request(PR) details such as pr author, pr title,
    pr body, the diff ulr, pr state and commit SHA for a given PR number"""
    pr = repo.get_pull(pr_num)
    return {
        "author": pr.user.login if pr.user else None,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "head_sha": pr.head.sha if pr.head else None,
    }


def get_commit_details(head_sha: str) -> List[dict[str, Any]]:
    """Returns the list of details of files changed. This includes, filename, status, additions,
    deletions, changes and patches. Takes in a head_sha as input"""
    commit = repo.get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append(
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            }
        )
    return changed_files


def get_file_content(file_path: str) -> str:
    """Returns the contents of a file from the repo"""
    content = repo.get_contents(file_path)
    if isinstance(content, list):
        return "\n".join(item.path for item in content)
    return content.decoded_content.decode("utf-8")


my_agent = ReActAgent(
    llm=llm,
    name="MyAgent",
    tools=[get_pr_details, get_commit_details, get_file_content],
)


async def main() -> None:
    query = input().strip()
    if not query:
        return

    prompt = RichPromptTemplate(query)
    context = Context(my_agent)

    handler = my_agent.run(prompt.format(), ctx=context)

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}", flush=True)
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print(event.response.content, flush=True)
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        git.close()