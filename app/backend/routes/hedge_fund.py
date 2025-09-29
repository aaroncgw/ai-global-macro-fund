from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import asyncio

from app.backend.database import get_db
from app.backend.models.schemas import ErrorResponse, HedgeFundRequest
from app.backend.models.events import StartEvent, ProgressUpdateEvent, ErrorEvent, CompleteEvent
from app.backend.services.graph import create_graph, parse_hedge_fund_response, run_graph_async
from app.backend.services.portfolio import create_portfolio
from app.backend.services.api_key_service import ApiKeyService

router = APIRouter(prefix="/hedge-fund")

@router.post(
    path="/run",
    responses={
        200: {"description": "Successful response with streaming updates"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def run(request_data: HedgeFundRequest, request: Request, db: Session = Depends(get_db)):
    """Run the hedge fund analysis with streaming updates."""
    try:
        # Hydrate API keys from database if not provided
        if not request_data.api_keys:
            api_key_service = ApiKeyService(db)
            request_data.api_keys = api_key_service.get_api_keys_dict()

        # Create the portfolio
        portfolio = create_portfolio(request_data.initial_cash, request_data.margin_requirement, request_data.tickers, request_data.portfolio_positions)

        # Construct agent graph using the React Flow graph structure
        graph = create_graph(
            graph_nodes=request_data.graph_nodes,
            graph_edges=request_data.graph_edges
        )
        graph = graph.compile()

        # Convert model_provider to string if it's an enum
        model_provider = request_data.model_provider
        if hasattr(model_provider, "value"):
            model_provider = model_provider.value

        # Set up streaming response
        async def event_generator():
            progress_queue = asyncio.Queue()
            graph_task = None
            disconnect_task = None

            # Global progress handler to capture individual agent updates
            def progress_handler(agent_name, ticker, status, analysis, timestamp):
                event = ProgressUpdateEvent(agent=agent_name, ticker=ticker, status=status, timestamp=timestamp, analysis=analysis)
                progress_queue.put_nowait(event)

            # Start the graph execution in a background task
            graph_task = asyncio.create_task(
                run_graph_async(
                    graph=graph,
                    portfolio=portfolio,
                    tickers=request_data.tickers,
                    model_name=request_data.model_name,
                    model_provider=model_provider,
                    selected_analysts=request_data.selected_analysts,
                    api_keys=request_data.api_keys
                )
            )
            
            # Start the disconnect detection task
            disconnect_task = asyncio.create_task(request.is_disconnected())

            # Send initial message
            yield StartEvent().to_sse()

            # Stream progress updates until graph_task completes or client disconnects
            while not graph_task.done():
                # Check if client disconnected
                if disconnect_task.done():
                    print("Client disconnected, cancelling graph execution")
                    graph_task.cancel()
                    try:
                        await graph_task
                    except asyncio.CancelledError:
                        pass
                    return

                # Check for progress updates
                try:
                    # Non-blocking get from queue
                    while not progress_queue.empty():
                        event = progress_queue.get_nowait()
                        yield event.to_sse()
                except asyncio.QueueEmpty:
                    pass

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            # Get the final result
            try:
                result = await graph_task
            except asyncio.CancelledError:
                print("Graph task was cancelled")
                return

            if not result:
                yield ErrorEvent(message="Failed to complete graph execution").to_sse()
                return

            # Send the final result
            yield CompleteEvent(
                message="Hedge fund analysis completed successfully",
                data=parse_hedge_fund_response(result)
            ).to_sse()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {str(e)}")

@router.get(
    path="/agents",
    responses={
        200: {"description": "List of available agents"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_agents():
    """Get list of available agents."""
    try:
        # This would need to be updated to return the new macro agents
        agents = [
            {
                "id": "macro_economist",
                "name": "Macro Economist",
                "description": "Analyzes macroeconomic indicators and trends"
            },
            {
                "id": "geopolitical_analyst", 
                "name": "Geopolitical Analyst",
                "description": "Assesses geopolitical risks and opportunities"
            },
            {
                "id": "correlation_specialist",
                "name": "Correlation Specialist", 
                "description": "Evaluates diversification benefits and portfolio balance"
            },
            {
                "id": "trader_agent",
                "name": "Trader Agent",
                "description": "Converts analysis into allocation proposals"
            },
            {
                "id": "risk_manager",
                "name": "Risk Manager",
                "description": "Adjusts allocations for risk factors"
            },
            {
                "id": "portfolio_optimizer",
                "name": "Portfolio Optimizer",
                "description": "Uses mathematical optimization for final allocations"
            }
        ]
        return {"agents": agents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching agents: {str(e)}")