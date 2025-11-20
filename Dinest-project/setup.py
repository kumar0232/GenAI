import argparse
import logging
import os

from data_utils import DataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment for the agent orchestrator"""
    os.makedirs("data", exist_ok=True)
    data_manager = DataManager(
        data_dir=os.getenv("DATA_DIR", "data"),
        supabase_url=os.getenv("SUPABASE_URL", "https://pkdcrqdwqyfrjenuwcls.supabase.co"),
        supabase_key=os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBrZGNycWR3cXlmcmplbnV3Y2xzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Njk2MjgzOSwiZXhwIjoyMDYyNTM4ODM5fQ.6BPvnnCp6ctiYrRwnbb5muGVHnIP1A3fEq5_teDNKfE")
    )
    logger.info("Loading knowledge base...")
    knowledge_base = data_manager.load_knowledge_base()
    logger.info("Preparing vector database...")
    vector_db = data_manager.prepare_vector_db(knowledge_base)
    logger.info("Environment setup complete")
    return data_manager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up the TechSolutions Agent Orchestrator environment"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the environment (delete existing data)",
    )
    args = parser.parse_args()

    if args.reset:
        logger.info("Resetting environment...")
        from supabase import create_client
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        for table in ["products", "technical", "conversations"]:
            supabase.table(table).delete().neq("id", "").execute()

    setup_environment()