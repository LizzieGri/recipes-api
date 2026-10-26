import os
import asyncio
from typing import Any
from dotenv import load_dotenv
from github import Github, Auth

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.openai import OpenAI

load_dotenv()

token = os.getenv("GITHUB_TOKEN")
repo_url = os.getenv("REPO_URL")

if not token or not repo_url:
    raise ValueError("No necessary values")

git = Github(auth=Auth.Token(token))

cleaned_url = repo_url.rstrip("/")
if cleaned_url.endswith(".git"):
    cleaned_url = cleaned_url[:-4]
parts = cleaned_url.split("/")
full_repo_name = f"{parts[-2]}/{parts[-1]}"

repo = git.get_repo(full_repo_name)
print(f"Got a connection: {full_repo_name}")

def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Get details about a pull request given its PR number, including author, title, body, diff_url, state, and commit SHAs."""
    pull_request = repo.get_pull(pr_number)
    commit_SHAs = [c.sha for c in pull_request.get_commits()]

    return {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "commit_SHAs": commit_SHAs,
    }


def get_file_contents(file_path: str) -> str:
    """Fetch the contents of a file from the repository given its file path."""
    try:
        file_content = repo.get_contents(file_path)
        return file_content.decoded_content.decode("utf-8")
    except Exception as e:
        return f"Error fetching file {file_path}: {str(e)}"


def get_commit_details(commit_sha: str) -> list[dict[str, Any]]:
    """Retrieve information about a commit given its SHA, such as changed files, status, additions, deletions, changes, and patch (diff)."""
    commit = repo.get_commit(commit_sha)
    changed_files = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files

pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
files_tool = FunctionTool.from_defaults(fn=get_file_contents)
commit_details_tool = FunctionTool.from_defaults(fn=get_commit_details)

tools = [pr_details_tool, files_tool, commit_details_tool]

system_prompt = """You are the context gathering agent. When gathering context, you MUST gather : 
-     
The details: author, title, body, diff_url, state, and head_sha; 
-     
Changed files; 
-     
Any requested for files; 
"""

llm = OpenAI(model="gpt-4o")

context_agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    system_prompt=system_prompt,
)


async def main():
    query = input().strip()
    prompt = RichPromptTemplate(query)

    handler = context_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print(event.response.content)
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")

if __name__ == "__main__":
    asyncio.run(main())
    git.close()