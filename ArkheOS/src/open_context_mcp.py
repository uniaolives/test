"""
Servidor MCP Open Context para Arkhe(n) OS.
Implementa a especificação Plurality Open Context para portabilidade de memória.
"""

from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional
import time
from arkhe.cortex_memory import CortexMemory

def create_open_context_server(cortex: CortexMemory):
    mcp = FastMCP("Plurality Open Context", host="0.0.0.0", port=8002)

    @mcp.tool()
    def get_user_memory_buckets() -> List[str]:
        """List all memory buckets (AI profiles) for the user."""
        return cortex.list_buckets()

    @mcp.tool()
    def list_items_in_memory_bucket(bucket_name: str) -> List[Dict[str, Any]]:
        """List stored items in a specific bucket (metadata only)."""
        return cortex.list_items(bucket_name)

    @mcp.tool()
    def search_memory(query: str, bucket_name: str = "arkhe_insights", n_results: int = 3) -> Dict[str, Any]:
        """Semantic search across buckets with relevance scoring."""
        results = cortex.recall(query, bucket_name=bucket_name, n_results=n_results)
        return results

    @mcp.tool()
    def read_context(bucket_name: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Read the full content of a stored item."""
        return cortex.read_item(bucket_name, item_id)

    @mcp.tool()
    def save_memory(content: str, topic: str, bucket_name: str = "arkhe_insights") -> str:
        """Save text content to a specific memory bucket."""
        item_id = cortex.memorize(
            topic=topic,
            summary=content,
            confidence=1.0,
            doc_id="mcp_input",
            bucket_name=bucket_name
        )
        return item_id

    @mcp.tool()
    def save_conversation(conversation: List[Dict[str, str]], topic: str, bucket_name: str = "arkhe_insights") -> str:
        """Save a conversation (chat history) to a memory bucket."""
        # Convert conversation list to a single string
        formatted_conv = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
        item_id = cortex.memorize(
            topic=topic,
            summary=formatted_conv,
            confidence=1.0,
            doc_id="mcp_conversation",
            bucket_name=bucket_name,
            related_nodes=["conversation_history"]
        )
        return item_id

    @mcp.tool()
    def create_memory_bucket(bucket_name: str) -> str:
        """Create a new memory bucket for organizing saved content."""
        return cortex.create_bucket(bucket_name)

    return mcp
