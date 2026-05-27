"""Tool descriptions exposed to the search-only OPTA agent."""


def convert_tools_to_description(tools: list[dict]) -> str:
    blocks = []
    for index, tool in enumerate(tools, start=1):
        fn = tool["function"]
        lines = [
            f"---- BEGIN FUNCTION #{index}: {fn['name']} ----",
            f"Description: {fn['description']}",
        ]
        properties = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        if properties:
            lines.append("Parameters:")
            for pos, (name, info) in enumerate(properties.items(), start=1):
                status = "required" if name in required else "optional"
                typ = info.get("type", "string")
                desc = info.get("description", "")
                lines.append(f"  ({pos}) {name} ({typ}, {status}): {desc}")
        else:
            lines.append("No parameters are required for this function.")
        lines.append(f"---- END FUNCTION #{index} ----")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def search_tool():
    search = {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the local retrieval server. Returns docids, URLs, and "
                "short document snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "topk": {
                        "type": "integer",
                        "description": "Number of results to show.",
                    },
                },
                "required": ["query"],
            },
        },
    }
    open_page = {
        "type": "function",
        "function": {
            "name": "open_page",
            "description": (
                "Open a retrieved page by docid or URL. Prefer docid values "
                "returned by search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "docid": {
                        "type": "string",
                        "description": "Document ID from a previous search result.",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL from a previous search result.",
                    },
                },
                "required": [],
            },
        },
    }
    finish = {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Succinct final answer.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief evidence-grounded explanation with docid citations.",
                    },
                    "confidence": {
                        "type": "string",
                        "description": "Confidence from 0% to 100%.",
                    },
                },
                "required": ["answer", "explanation"],
            },
        },
    }
    return [search, open_page, finish]
