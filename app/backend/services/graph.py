import asyncio
import json
import re
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.backend.services.agent_service import create_agent_function
from src.graph.state import AgentState

# Macro agent configuration for the new system
MACRO_AGENT_CONFIG = {
    "macro_economist": {
        "display_name": "Macro Economist",
        "description": "Analyzes macroeconomic indicators and trends",
        "agent_func": "macro_economist_agent",
        "type": "analyst",
        "order": 0,
    },
    "geopolitical_analyst": {
        "display_name": "Geopolitical Analyst", 
        "description": "Assesses geopolitical risks and opportunities",
        "agent_func": "geopolitical_analyst_agent",
        "type": "analyst",
        "order": 1,
    },
    "correlation_specialist": {
        "display_name": "Correlation Specialist",
        "description": "Evaluates diversification benefits and portfolio balance", 
        "agent_func": "correlation_specialist_agent",
        "type": "analyst",
        "order": 2,
    },
    "trader_agent": {
        "display_name": "Trader Agent",
        "description": "Converts analysis into allocation proposals",
        "agent_func": "trader_agent",
        "type": "trader",
        "order": 3,
    },
    "risk_manager": {
        "display_name": "Risk Manager",
        "description": "Adjusts allocations for risk factors",
        "agent_func": "risk_manager_agent", 
        "type": "risk",
        "order": 4,
    },
    "portfolio_optimizer": {
        "display_name": "Portfolio Optimizer",
        "description": "Uses mathematical optimization for final allocations",
        "agent_func": "portfolio_optimizer_agent",
        "type": "optimizer",
        "order": 5,
    }
}

def extract_base_agent_key(unique_id: str) -> str:
    """
    Extract the base agent key from a unique node ID.
    
    Args:
        unique_id: The unique node ID with suffix (e.g., "macro_economist_abc123")
        
    Returns:
        The base agent key (e.g., "macro_economist")
    """
    # Remove the suffix after the last underscore
    parts = unique_id.split('_')
    if len(parts) > 1:
        # Check if the last part is a suffix (alphanumeric)
        last_part = parts[-1]
        if last_part.isalnum() and len(last_part) > 2:
            return '_'.join(parts[:-1])
    return unique_id

def create_graph(graph_nodes, graph_edges):
    """
    Create a LangGraph from the React Flow graph structure.
    
    Args:
        graph_nodes: List of node objects from React Flow
        graph_edges: List of edge objects from React Flow
        
    Returns:
        A compiled LangGraph StateGraph
    """
    # Create the state graph
    graph = StateGraph(AgentState)
    
    # Add nodes to the graph
    for node in graph_nodes:
        node_id = node["id"]
        node_type = node["type"]
        
        if node_type == "start":
            # Start node - just pass through
            def start_node(state):
                return state
            graph.add_node(node_id, start_node)
            
        elif node_type == "agent":
            # Agent node - create agent function
            agent_key = extract_base_agent_key(node_id)
            agent_config = MACRO_AGENT_CONFIG.get(agent_key)
            
            if agent_config:
                agent_func = create_agent_function(
                    agent_key=agent_key,
                    agent_config=agent_config,
                    node_id=node_id
                )
                graph.add_node(node_id, agent_func)
            else:
                # Fallback for unknown agents
                def fallback_agent(state):
                    return state
                graph.add_node(node_id, fallback_agent)
                
        elif node_type == "output":
            # Output node - final processing
            def output_node(state):
                return state
            graph.add_node(node_id, output_node)
    
    # Add edges to the graph
    for edge in graph_edges:
        source = edge["source"]
        target = edge["target"]
        
        if target == "END":
            graph.add_edge(source, END)
        else:
            graph.add_edge(source, target)
    
    # Set entry point (first start node)
    start_nodes = [node for node in graph_nodes if node["type"] == "start"]
    if start_nodes:
        graph.set_entry_point(start_nodes[0]["id"])
    
    return graph

async def run_graph_async(graph, portfolio, tickers, model_name, model_provider, selected_analysts, api_keys):
    """
    Run the graph asynchronously with the given parameters.
    
    Args:
        graph: The compiled LangGraph
        portfolio: Portfolio object
        tickers: List of ticker symbols
        model_name: LLM model name
        model_provider: LLM provider
        selected_analysts: List of selected analyst IDs
        api_keys: Dictionary of API keys
        
    Returns:
        The final state from the graph execution
    """
    try:
        # Create initial state
        initial_state = {
            "portfolio": portfolio,
            "tickers": tickers,
            "model_name": model_name,
            "model_provider": model_provider,
            "selected_analysts": selected_analysts,
            "api_keys": api_keys,
            "messages": [],
            "current_step": 0,
            "total_steps": len(selected_analysts) if selected_analysts else 0
        }
        
        # Run the graph
        result = await graph.ainvoke(initial_state)
        return result
        
    except Exception as e:
        print(f"Error running graph: {e}")
        return None

def parse_hedge_fund_response(result):
    """
    Parse the hedge fund response into a standardized format.
    
    Args:
        result: The result from the graph execution
        
    Returns:
        Parsed response data
    """
    if not result:
        return {"error": "No result from graph execution"}
    
    # Extract key information from the result
    parsed = {
        "portfolio_value": result.get("portfolio", {}).get("total_value", 0),
        "allocations": result.get("allocations", {}),
        "messages": result.get("messages", []),
        "final_allocations": result.get("final_allocations", {}),
        "agent_reasoning": result.get("agent_reasoning", {})
    }
    
    return parsed