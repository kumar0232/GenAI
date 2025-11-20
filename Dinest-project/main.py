import logging
import os
from typing import Any, Dict, List, Optional
import asyncio
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="TechSolutions Support Agent Orchestrator")

# Global variables for lazy initialization
data_manager = None
llm = None
agent_orchestrator = None
initialization_error = None

# Define API models
class CustomerQuery(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    agent: str
    conversation_id: str

async def initialize_system():
    """Initialize the system components lazily with detailed error handling"""
    global data_manager, llm, agent_orchestrator, initialization_error
    
    if agent_orchestrator is not None:
        logger.info("System already initialized")
        return True
    
    if initialization_error is not None:
        logger.error(f"Previous initialization failed: {initialization_error}")
        return False
    
    try:
        logger.info("Starting system initialization...")
        
        # Check environment variables
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            logger.warning("GROQ_API_KEY not set, using default (may not work)")
        
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_KEY", "")
        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not set, will use fallback mode")
        
        # Create sample data first
        logger.info("Creating sample data if needed...")
        await create_sample_data_if_needed()
        
        # Import modules with error handling
        try:
            logger.info("Importing data_utils...")
            from data_utils import DataManager
            logger.info("data_utils imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import DataManager: {e}")
            initialization_error = f"Import error: {e}"
            return False
        
        try:
            logger.info("Importing agent_implementations...")
            from agent_implementations import LLMUtils, AgentOrchestrator
            logger.info("agent_implementations imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import agent implementations: {e}")
            initialization_error = f"Import error: {e}"
            return False
        
        # Initialize Data Manager
        logger.info("Initializing DataManager...")
        try:
            data_manager = DataManager(
                data_dir=os.getenv("DATA_DIR", "data"),
                supabase_url=supabase_url,
                supabase_key=supabase_key
            )
            logger.info("DataManager initialized successfully")
        except Exception as e:
            logger.error(f"DataManager initialization failed: {e}")
            logger.error(traceback.format_exc())
            initialization_error = f"DataManager error: {e}"
            return False
        
        # Load knowledge base
        logger.info("Loading knowledge base...")
        try:
            knowledge_base = data_manager.load_knowledge_base()
            logger.info("Knowledge base loaded successfully")
        except Exception as e:
            logger.error(f"Knowledge base loading failed: {e}")
            logger.error(traceback.format_exc())
            initialization_error = f"Knowledge base error: {e}"
            return False
        
        # Prepare vector database
        logger.info("Preparing vector database...")
        try:
            vector_db = data_manager.prepare_vector_db(knowledge_base)
            logger.info("Vector database prepared successfully")
        except Exception as e:
            logger.error(f"Vector database preparation failed: {e}")
            logger.error(traceback.format_exc())
            initialization_error = f"Vector DB error: {e}"
            return False
        
        # Initialize LLM utils
        logger.info("Initializing LLM utils...")
        try:
            llm = LLMUtils()
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            logger.error(traceback.format_exc())
            initialization_error = f"LLM error: {e}"
            return False
        
        # Initialize Agent Orchestrator
        logger.info("Initializing Agent Orchestrator...")
        try:
            agent_orchestrator = AgentOrchestrator(
                llm_utils=llm,
                knowledge_base=knowledge_base,
                vector_db=vector_db
            )
            logger.info("Agent Orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Agent Orchestrator initialization failed: {e}")
            logger.error(traceback.format_exc())
            initialization_error = f"Agent Orchestrator error: {e}"
            return False
        
        logger.info("System initialization completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"Unexpected error during initialization: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        initialization_error = error_msg
        return False

async def create_sample_data_if_needed():
    """Create sample data files if they don't exist"""
    try:
        data_dir = os.getenv("DATA_DIR", "data")
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Data directory: {data_dir}")
        
        # Sample product catalog
        product_catalog_path = os.path.join(data_dir, "product_catalog.json")
        if not os.path.exists(product_catalog_path):
            logger.info("Creating sample product catalog...")
            sample_catalog = {
                "products": [
                    {
                        "id": "cm-pro",
                        "name": "CloudManager Pro",
                        "description": "Professional cloud management solution",
                        "price": {"monthly": 149.99, "annual": 1499.99},
                        "features": [
                            {"name": "Multi-cloud support", "description": "Support for AWS, Azure, GCP"},
                            {"name": "Advanced monitoring", "description": "Real-time monitoring and alerts"}
                        ],
                        "limitations": ["Limited to 100 instances"],
                        "target_audience": "Medium businesses"
                    },
                    {
                        "id": "cm-enterprise",
                        "name": "CloudManager Enterprise",
                        "description": "Enterprise-grade cloud management",
                        "price": {"monthly": 499.99, "annual": 4999.99},
                        "features": [
                            {"name": "Unlimited instances", "description": "No limits on managed instances"},
                            {"name": "24/7 support", "description": "Round-the-clock premium support"}
                        ],
                        "limitations": [],
                        "target_audience": "Large enterprises"
                    }
                ],
                "addons": [
                    {
                        "id": "addon-premium-support",
                        "name": "Premium Support",
                        "description": "24/7 premium support with dedicated account manager",
                        "price": 299.99,
                        "details": "Response time within 1 hour"
                    }
                ],
                "bundles": [
                    {
                        "id": "bundle-complete",
                        "name": "Complete Solution Bundle",
                        "description": "CloudManager Pro with Premium Support",
                        "included_products": ["cm-pro", "addon-premium-support"],
                        "price": {"monthly": 399.99, "annual": 3999.99, "saving_percentage": 15}
                    }
                ]
            }
            
            import json
            with open(product_catalog_path, "w") as f:
                json.dump(sample_catalog, f, indent=2)
            logger.info("Created sample product catalog")
        
        # Sample FAQ
        faq_path = os.path.join(data_dir, "faq.json")
        if not os.path.exists(faq_path):
            logger.info("Creating sample FAQ...")
            sample_faq = {
                "categories": [
                    {
                        "name": "General",
                        "questions": [
                            {
                                "question": "What is CloudManager?",
                                "answer": "CloudManager is a comprehensive cloud management platform that helps businesses manage their multi-cloud infrastructure efficiently."
                            },
                            {
                                "question": "How do I get started?",
                                "answer": "You can start with our free trial. Simply sign up on our website and follow the setup wizard."
                            }
                        ]
                    },
                    {
                        "name": "Billing",
                        "questions": [
                            {
                                "question": "What payment methods do you accept?",
                                "answer": "We accept all major credit cards, PayPal, and bank transfers for enterprise customers."
                            }
                        ]
                    }
                ]
            }
            
            import json
            with open(faq_path, "w") as f:
                json.dump(sample_faq, f, indent=2)
            logger.info("Created sample FAQ")
        
        # Sample technical documentation
        tech_docs_path = os.path.join(data_dir, "tech_documentation.md")
        if not os.path.exists(tech_docs_path):
            logger.info("Creating sample technical documentation...")
            sample_tech_docs = """# TechSolutions Technical Documentation

## Getting Started

### Installation
1. Download the CloudManager installer
2. Run the installer with administrator privileges
3. Follow the setup wizard

### Configuration
- Configure your cloud provider credentials
- Set up monitoring preferences
- Configure alert thresholds

## Troubleshooting

### Common Issues

#### Error E1234: API Connection Failure
**Symptoms:** Unable to connect to cloud provider APIs

**Solutions:**
1. Verify API credentials in Settings > Connections
2. Check firewall settings
3. Ensure cloud provider services are operational

#### Error E5678: Container Image Verification Failed
**Symptoms:** Container deployment fails with verification error

**Solutions:**
1. Check image integrity and re-pull from registry
2. Verify signature configuration
3. Review scan results in Security tab

### Performance Issues
- Monitor CPU and memory usage
- Check network connectivity
- Review log files for errors
"""
            
            with open(tech_docs_path, "w") as f:
                f.write(sample_tech_docs)
            logger.info("Created sample technical documentation")
        
        # Sample customer conversations
        conversations_path = os.path.join(data_dir, "customer_conversations.jsonl")
        if not os.path.exists(conversations_path):
            logger.info("Creating sample customer conversations...")
            sample_conversations = [
                {
                    "conversation_id": "conv-001",
                    "customer_email": "user@example.com",
                    "agent_name": "Agent Smith",
                    "messages": [
                        {"role": "customer", "content": "I'm having trouble connecting to AWS"},
                        {"role": "agent", "content": "I can help you with that. Let me check your configuration."},
                        {"role": "customer", "content": "Thank you, I appreciate your help"}
                    ]
                }
            ]
            
            import json
            with open(conversations_path, "w") as f:
                for conv in sample_conversations:
                    f.write(json.dumps(conv) + "\n")
            logger.info("Created sample customer conversations")
        
        logger.info("Sample data creation completed")
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        logger.error(traceback.format_exc())
        raise

@app.post("/api/query", response_model=AgentResponse)
async def process_customer_query(query: CustomerQuery):
    """Process a customer support query"""
    try:
        logger.info(f"Processing query: {query.query[:100]}...")
        
        # Initialize system if not already done
        initialization_success = await initialize_system()
        if not initialization_success:
            error_msg = f"System initialization failed: {initialization_error}"
            logger.error(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)
            
        if not agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        logger.info("Processing query with agent orchestrator...")
        result = await agent_orchestrator.process_query(query.query, query.conversation_id)
        logger.info("Query processed successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing query: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Simple fallback query processor for when full system fails
@app.post("/api/query/simple")
async def process_simple_query(query: CustomerQuery):
    """Simple fallback query processor that doesn't require full initialization"""
    try:
        # Simple pattern matching responses
        query_text = query.query.lower()
        
        if any(word in query_text for word in ["price", "cost", "pricing", "how much"]):
            response = """Our pricing is as follows:
- CloudManager Pro: $149.99/month or $1,499.99/year
- CloudManager Enterprise: $499.99/month or $4,999.99/year
- Premium Support Add-on: $299.99/month

For more detailed information, please visit our pricing page or contact our sales team."""
        
        elif any(word in query_text for word in ["error", "issue", "problem", "troubleshoot"]):
            response = """For technical issues, please try these general steps:
1. Check your internet connection
2. Verify your credentials are correct
3. Review the error logs for specific error codes
4. Restart the application if needed

If the problem persists, please contact our technical support team with the specific error message."""
        
        elif any(word in query_text for word in ["account", "user", "login", "access"]):
            response = """For account-related questions:
- To add users: Go to Settings > User Management
- To reset passwords: Use the 'Forgot Password' link on the login page
- To upgrade your plan: Visit the Billing section in your account
- For access issues: Contact our support team

Our support team is available 24/7 for Enterprise customers."""
        
        else:
            response = """Thank you for contacting TechSolutions support. 

I'm currently operating in simplified mode. For the best assistance, please:
1. Check our FAQ section on the website
2. Review our documentation at docs.techsolutions.example.com
3. Contact our support team directly for complex issues

Is there anything specific I can help you with regarding our CloudManager products?"""
        
        return AgentResponse(
            response=response,
            agent="simple_fallback",
            conversation_id=query.conversation_id or "simple_session"
        )
        
    except Exception as e:
        logger.error(f"Error in simple query processor: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug/initialization")
async def debug_initialization():
    """Debug endpoint to check initialization status"""
    try:
        # Try to initialize and return detailed status
        success = await initialize_system()
        
        return {
            "initialization_successful": success,
            "initialization_error": initialization_error,
            "components": {
                "data_manager": data_manager is not None,
                "llm": llm is not None,
                "agent_orchestrator": agent_orchestrator is not None
            },
            "environment": {
                "data_dir": os.getenv("DATA_DIR", "data"),
                "data_dir_exists": os.path.exists(os.getenv("DATA_DIR", "data")),
                "has_groq_key": bool(os.getenv("GROQ_API_KEY", "")),
                "has_supabase_url": bool(os.getenv("SUPABASE_URL", "")),
                "has_supabase_key": bool(os.getenv("SUPABASE_KEY", ""))
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Mock API endpoints for testing
@app.get("/api/orders/{order_id}")
async def get_order(order_id: str):
    """Mock Order API endpoint"""
    orders = {
        "ORD-12345": {
            "order_id": "ORD-12345",
            "status": "shipped",
            "items": [{"product_id": "cm-pro", "quantity": 1, "price": 149.99}],
            "total": 149.99,
            "order_date": "2023-09-10",
            "shipping_date": "2023-09-12",
            "delivery_date": "2023-09-15",
        },
        "ORD-56789": {
            "order_id": "ORD-56789",
            "status": "processing",
            "items": [
                {"product_id": "cm-enterprise", "quantity": 1, "price": 499.99},
                {"product_id": "addon-premium-support", "quantity": 1, "price": 299.99},
            ],
            "total": 799.98,
            "order_date": "2023-09-22",
            "shipping_date": None,
            "delivery_date": None,
        },
    }
    if order_id in orders:
        return orders[order_id]
    else:
        raise HTTPException(status_code=404, detail="Order not found")

@app.get("/api/accounts/{account_id}")
async def get_account(account_id: str):
    """Mock Account API endpoint"""
    accounts = {
        "ACC-1111": {
            "account_id": "ACC-1111",
            "name": "Acme Corp",
            "subscription": {
                "plan": "cm-pro",
                "status": "active",
                "start_date": "2023-01-15",
                "renewal_date": "2024-01-15",
                "payment_method": "credit_card",
                "auto_renew": True,
            },
            "users": [
                {"email": "admin@acme.example.com", "role": "admin"},
                {"email": "user1@acme.example.com", "role": "viewer"},
                {"email": "user2@acme.example.com", "role": "operator"},
            ],
        },
    }
    if account_id in accounts:
        return accounts[account_id]
    else:
        raise HTTPException(status_code=404, detail="Account not found")

@app.post("/api/diagnose")
async def diagnose_issue(request: Request):
    """Mock troubleshooting API endpoint"""
    try:
        data = await request.json()
        issue_description = data.get("description", "")
        if "error e1234" in issue_description.lower():
            return {
                "issue_id": "E1234",
                "name": "API Connection Failure",
                "solutions": [
                    "Verify API credentials in Settings > Connections",
                    "Check if your firewall allows outbound connections",
                    "Ensure cloud provider services are operational",
                ],
                "documentation_link": "docs.techsolutions.example.com/errors/e1234",
            }
        else:
            return {
                "issue_id": "unknown",
                "name": "General Issue",
                "solutions": [
                    "Check application logs for specific error messages",
                    "Verify your configuration settings",
                    "Contact support with error details",
                ],
                "documentation_link": "docs.techsolutions.example.com/troubleshooting",
            }
    except Exception as e:
        logger.error(f"Error in diagnose endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Support Agent Orchestrator")
    logger.info("System will initialize on first request to avoid blocking startup")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Support Agent Orchestrator")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "initialized": agent_orchestrator is not None,
        "initialization_error": initialization_error
    }

@app.get("/")
async def root():
    return {
        "message": "TechSolutions Support Agent Orchestrator is running",
        "endpoints": {
            "main_query": "/api/query",
            "simple_query": "/api/query/simple",
            "debug": "/api/debug/initialization",
            "health": "/health"
        }
    }

@app.get("/api/status")
async def get_system_status():
    return {
        "status": "running",
        "initialized": agent_orchestrator is not None,
        "initialization_error": initialization_error,
        "components": {
            "data_manager": data_manager is not None,
            "llm": llm is not None,
            "agent_orchestrator": agent_orchestrator is not None
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)